from LLM_PCT import (
    create_statements,
    create_scores,
    take_pct_test,
    get_all_results,
    display_results,
    PCTPrompts,
)
import os
from urllib.parse import urlparse, parse_qs

# CONFIG

device = -1  # -1 for CPU, 0 for GPU
threshold = 0.5
pct_asset_path = os.path.join(os.getcwd(), "pct-assets")
pct_result_path = os.path.join(os.getcwd(), "pct-assets", "results")


print("=====================================")
print("*** RESULTS ***")

political_beliefs = get_all_results(pct_result_path)

ones_we_care_about = {}
for k in political_beliefs.keys():
    if "base_" in k or "IH" in k:
        ones_we_care_about[k] = political_beliefs[k]

# create PCT plot for all findings and save as shareable/embeddable url
results_url = display_results(ones_we_care_about)
print(results_url)
