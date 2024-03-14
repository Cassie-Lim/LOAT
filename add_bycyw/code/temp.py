import json

unseen_path = "leaderboard_jsons/tests_actseqs_seq_low_seqsearch_dropFailLoc_unseen.json"
seen_path = "leaderboard_jsons/tests_actseqs_seq_low_seqsearch_dropFailLoc_seen.json"

unseen_data = json.load(open(unseen_path, "r"))
seen_data = json.load(open(seen_path, "r"))

total_data = {}
total_data["tests_seen"] = seen_data["tests_seen"]
total_data["tests_unseen"] = unseen_data["tests_unseen"]

leaderboard_file = "leaderboard_jsons"
file_name = "tests_actseqs_seq_low_seqsearch_dropFailLoc.json"
with open(leaderboard_file + "/" + file_name, "w") as f:
    json.dump(total_data, f,indent=4)
