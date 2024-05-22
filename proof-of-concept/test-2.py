import os

from LLM_PCT import (
    create_scores,
    take_pct_test
)

pct_asset_path = os.path.join(os.getcwd(), "pct-assets")

print('*** PCT test for copilot_auth_right')
take_pct_test(pct_assets_path=pct_asset_path,
              model="copilot_auth_right", threshold=0.5)

print('*** PCT test for copilot_auth_left')
take_pct_test(pct_assets_path=pct_asset_path,
              model="copilot_auth_left", threshold=0.5)

print('*** PCT test for copilot_lib_right')
take_pct_test(pct_assets_path=pct_asset_path,
              model="copilot_lib_right", threshold=0.5)

print('*** PCT test for copilot_lib_left')
take_pct_test(pct_assets_path=pct_asset_path,
              model="copilot_lib_left", threshold=0.5)
