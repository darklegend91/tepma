import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import facts


DELHI_RECORDS = (
    {
        "circlename": "Delhi Circle",
        "regionname": "DivReportingCircle",
        "divisionname": "New Delhi Central Division",
        "officename": "Connaught Place SO",
        "pincode": "110001",
        "officetype": "PO",
        "delivery": "Non Delivery",
        "district": "NEW DELHI",
        "statename": "DELHI",
    },
    {
        "circlename": "Delhi Circle",
        "regionname": "DivReportingCircle",
        "divisionname": "New Delhi GPO Division",
        "officename": "New Delhi GPO",
        "pincode": "110001",
        "officetype": "HO",
        "delivery": "Delivery",
        "district": "NEW DELHI",
        "statename": "DELHI",
    },
)


class WithoutLocalPincodeDatabase:
    def setUp(self):
        super().setUp()
        self.local_database_patch = patch.object(
            facts,
            "_fetch_local_post_offices",
            side_effect=facts.PincodeDatabaseUnavailable("not installed"),
        )
        self.local_database_patch.start()

    def tearDown(self):
        self.local_database_patch.stop()
        super().tearDown()


class PincodeLookupTests(WithoutLocalPincodeDatabase, unittest.TestCase):
    def test_fetches_only_matching_records_from_department_of_posts_api(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "status": "ok",
            "records": [
                DELHI_RECORDS[0],
                {**DELHI_RECORDS[0], "pincode": "110002"},
                "malformed",
            ],
        }
        facts._fetch_post_offices.cache_clear()
        try:
            with patch.object(facts.httpx, "get", return_value=response) as get:
                records = facts._fetch_post_offices("110001")
        finally:
            facts._fetch_post_offices.cache_clear()

        self.assertEqual(records, (DELHI_RECORDS[0],))
        params = get.call_args.kwargs["params"]
        self.assertEqual(params["filters[pincode]"], "110001")
        self.assertNotIn("Connaught Place", str(get.call_args))

    def test_secondary_api_schema_is_converted_for_the_same_matcher(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = [{
            "Status": "Success",
            "PostOffice": [{
                "Name": "Connaught Place",
                "Pincode": "110001",
                "District": "New Delhi",
                "State": "Delhi",
                "Division": "New Delhi Central",
                "Circle": "Delhi",
                "Region": "Delhi",
                "BranchType": "Sub Post Office",
                "DeliveryStatus": "Non-Delivery",
            }],
        }]
        facts._fetch_secondary_post_offices.cache_clear()
        try:
            with patch.object(facts.httpx, "get", return_value=response):
                records = facts._fetch_secondary_post_offices("110001")
        finally:
            facts._fetch_secondary_post_offices.cache_clear()

        self.assertEqual(records[0]["officename"], "Connaught Place")
        self.assertEqual(records[0]["statename"], "Delhi")

    def test_valid_pin_matches_post_office_and_place(self):
        with patch.object(facts, "_fetch_post_offices", return_value=DELHI_RECORDS):
            result = facts.lookup_pincode("Connaught Place, 110001")

        self.assertIsNotNone(result)
        self.assertTrue(result["verified"])
        self.assertTrue(result["place_match"])
        self.assertEqual(result["match_field"], "officename")
        self.assertEqual(result["district"], "New Delhi")
        self.assertEqual(result["state"], "Delhi")
        self.assertEqual(result["source"], "department_of_posts_data_gov_in")

    def test_valid_pin_with_wrong_place_is_not_treated_as_a_match(self):
        with patch.object(facts, "_fetch_post_offices", return_value=DELHI_RECORDS):
            result = facts.lookup_pincode("Mumbai, Maharashtra 110001")

        self.assertIsNotNone(result)
        self.assertTrue(result["verified"])
        self.assertFalse(result["place_match"])
        self.assertIsNone(result["matched_place"])

    def test_api_not_found_is_invalid_and_does_not_use_prefix_fallback(self):
        with patch.object(facts, "_fetch_post_offices", return_value=()):
            result = facts.lookup_pincode("New Delhi 119999")

        self.assertIsNone(result)

    def test_service_failure_uses_unverified_prefix_fallback(self):
        with (
            patch.object(
                facts,
                "_fetch_post_offices",
                side_effect=facts.PincodeServiceUnavailable("offline"),
            ),
            patch.object(
                facts,
                "_fetch_secondary_post_offices",
                side_effect=facts.PincodeServiceUnavailable("offline"),
            ),
            patch.object(facts, "pin_ranges", return_value={"11": "Delhi"}),
        ):
            result = facts.lookup_pincode("New Delhi 110001")

        self.assertIsNotNone(result)
        self.assertFalse(result["verified"])
        self.assertEqual(result["state"], "Delhi")
        self.assertEqual(result["source"], "local_prefix_fallback")

    def test_official_timeout_uses_verified_secondary_records(self):
        with (
            patch.object(
                facts,
                "_fetch_post_offices",
                side_effect=facts.PincodeServiceUnavailable("offline"),
            ),
            patch.object(
                facts,
                "_fetch_secondary_post_offices",
                return_value=DELHI_RECORDS,
            ),
        ):
            result = facts.lookup_pincode("Connaught Place 110001")

        self.assertIsNotNone(result)
        self.assertTrue(result["verified"])
        self.assertTrue(result["place_match"])
        self.assertEqual(result["confidence"], 0.95)
        self.assertEqual(result["source"], "postalpincode_in_fallback")

    def test_only_pin_is_valid_but_has_no_place_match_decision(self):
        with patch.object(facts, "_fetch_post_offices", return_value=DELHI_RECORDS):
            result = facts.lookup_pincode("110001")

        self.assertIsNotNone(result)
        self.assertTrue(result["verified"])
        self.assertIsNone(result["place_match"])


class LocalPincodeDatabaseTests(unittest.TestCase):
    def test_local_database_is_preferred_without_an_api_request(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "pincodes.sqlite3"
            with sqlite3.connect(database) as connection:
                connection.execute(
                    """
                    CREATE TABLE post_offices (
                        pincode TEXT, officename TEXT, district TEXT, statename TEXT,
                        divisionname TEXT, circlename TEXT, regionname TEXT,
                        officetype TEXT, delivery TEXT
                    )
                    """
                )
                connection.execute(
                    "INSERT INTO post_offices VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        "110001", "Connaught Place SO", "NEW DELHI", "DELHI",
                        "New Delhi Central Division", "Delhi Circle", "Delhi Region",
                        "PO", "Non Delivery",
                    ),
                )

            facts._fetch_local_post_offices.cache_clear()
            try:
                with (
                    patch.object(facts, "PINCODE_DB_PATH", database),
                    patch.object(
                        facts,
                        "_fetch_post_offices",
                        side_effect=AssertionError("API must not be called"),
                    ),
                ):
                    result = facts.lookup_pincode("Connaught Place 110001")
            finally:
                facts._fetch_local_post_offices.cache_clear()

        self.assertIsNotNone(result)
        self.assertTrue(result["place_match"])
        self.assertEqual(result["source"], "local_department_of_posts_snapshot")


class PincodeApplicationTests(WithoutLocalPincodeDatabase, unittest.TestCase):
    def test_verified_match_adds_district_and_state(self):
        with patch.object(facts, "_fetch_post_offices", return_value=DELHI_RECORDS):
            profile = facts.apply_facts({"location": "Connaught Place 110001"})

        self.assertEqual(
            profile["location"],
            "Connaught Place 110001, New Delhi, Delhi",
        )
        self.assertEqual(
            profile["_corrections"][0]["source"],
            "department_of_posts_data_gov_in",
        )
        self.assertNotIn("_validation_warnings", profile)

    def test_mismatch_is_warned_about_but_never_rewritten(self):
        original = "Mumbai, Maharashtra 110001"
        with patch.object(facts, "_fetch_post_offices", return_value=DELHI_RECORDS):
            profile = facts.apply_facts({"location": original})

        self.assertEqual(profile["location"], original)
        self.assertNotIn("_corrections", profile)
        warning = profile["_validation_warnings"][0]
        self.assertEqual(warning["code"], "pincode_place_not_matched")
        self.assertEqual(warning["expected_state"], "Delhi")


class EmailNormalisationTests(unittest.TestCase):
    """Cases are real model outputs captured while benchmarking, plus raw spoken forms."""

    def test_spoken_addresses_become_real_ones(self):
        for spoken, expected in (
            ("aditya dot p at gee mail dot com", "aditya.p@gmail.com"),
            ("simran dot kaur at yahoo dot co dot in", "simran.kaur@yahoo.co.in"),
            ("rahul underscore verma at outlook dot com", "rahul_verma@outlook.com"),
            ("priya at the rate of hot mail dot com", "priya@hotmail.com"),
            ("amit dash kumar at rediff mail dot com", "amit-kumar@rediffmail.com"),
        ):
            with self.subTest(spoken=spoken):
                self.assertEqual(facts.normalise_email(spoken), expected)

    def test_valid_looking_addresses_that_spell_their_own_punctuation(self):
        for garbled, expected in (
            ("rahul_underscore_verma@outlook.com", "rahul_verma@outlook.com"),
            ("aditya.dot.p@gee.mail.com", "aditya.p@gmail.com"),
            ("aditya.p@ gmail.com", "aditya.p@gmail.com"),
        ):
            with self.subTest(garbled=garbled):
                self.assertEqual(facts.normalise_email(garbled), expected)

    def test_good_addresses_are_never_touched(self):
        # "nathan"/"attar" contain "at", "dotto"/"dotmail" contain "dot".
        for address in ("already.valid@example.org", "nathan.kate@example.com",
                        "dotto.attar@dotmail.com", "a_b@c.co", "aditya@gmail.com"):
            with self.subTest(address=address):
                self.assertEqual(facts.normalise_email(address), address)

    def test_unrepairable_input_is_returned_unchanged_and_warned_about(self):
        self.assertEqual(facts.normalise_email("complete nonsense here"),
                         "complete nonsense here")
        profile = facts.apply_facts({"email": "complete nonsense here"})
        self.assertEqual(profile["email"], "complete nonsense here")
        self.assertEqual(profile["_validation_warnings"][0]["code"], "email_not_valid")

    def test_a_repair_is_recorded_as_a_correction(self):
        profile = facts.apply_facts({"email": "aditya dot p at gee mail dot com"})
        self.assertEqual(profile["email"], "aditya.p@gmail.com")
        self.assertEqual(profile["_corrections"][0]["field"], "email")
        self.assertNotIn("_validation_warnings", profile)


class InstitutionCorrectionTests(unittest.TestCase):
    """The reference file is not in git, so these also assert it is actually present."""

    def test_reference_file_is_installed(self):
        self.assertTrue(facts.institutions(),
                        "data/reference/universities.json is missing or empty")

    def test_speech_errors_are_corrected(self):
        for spoken, expected in (
            ("Thapadi University", "Thapar Institute of Engineering and Technology"),
            ("Guru Nanak Dev Universty", "Guru Nanak Dev University"),
            ("Panjab Univarsity", "Panjab University"),
            ("Delhi Technological Univercity", "Delhi Technological University"),
        ):
            with self.subTest(spoken=spoken):
                self.assertEqual(facts.correct_institution(spoken)[0], expected)

    def test_spelled_out_and_colloquial_names(self):
        for spoken, expected in (
            ("I I T Rurkee", "Indian Institute of Technology Roorkee"),
            ("NIT Trichy", "National Institute of Technology Tiruchirappalli"),
            ("IIT KGP", "Indian Institute of Technology Kharagpur"),
            ("BITS Pilani", "Birla Institute of Technology and Science, Pilani"),
        ):
            with self.subTest(spoken=spoken):
                self.assertEqual(facts.correct_institution(spoken)[0], expected)

    def test_a_shared_common_word_is_not_enough_to_match(self):
        # "Guru" alone must not pull this towards Guru Gobind Singh Indraprastha.
        self.assertEqual(
            facts.correct_institution("Guru Nanak Dev University")[0],
            "Guru Nanak Dev University",
        )

    def test_unknown_institutions_are_never_invented(self):
        for spoken in ("Some Random Unknown College",
                       "Shri Ram Institute of Rural Development"):
            with self.subTest(spoken=spoken):
                self.assertEqual(facts.correct_institution(spoken)[0], spoken)

    def test_a_missing_reference_file_does_not_abort_the_interview(self):
        try:
            with patch.object(facts, "_institutions", None), \
                 patch.object(facts, "REF_DIR",
                              Path(tempfile.gettempdir()) / "tepma-absent"):
                self.assertEqual(facts.institutions(), [])
                profile = facts.apply_facts({
                    "education": [{"degree": "B.Tech", "institution": "Thapadi University"}]})
            self.assertEqual(profile["education"][0]["institution"], "Thapadi University")
        finally:
            facts._institutions = None


if __name__ == "__main__":
    unittest.main()
