from pkg import first_clean
from pathlib import Path
import pandas as pd
import fxs  

if __name__ == "__main__":
    BASE = Path("/path/to/data_root")   # parent dir that contains per-country folders
    allcountries = first_clean(BASE)

    results = []
    failed_countries = []

    for c in allcountries:
        try:
            fxs.start = str(BASE / c)  # set module-global 'start' that fxs.py expects
            print(f"Running {c} with start={fxs.start}")

            final = fxs.run_like_hocker()  # returns a DataFrame or None

            if final is None or (hasattr(final, "empty") and final.empty):
                print(f"[WARN] {c}: got None/empty result, skipping.")
                failed_countries.append(c)
                continue

            # add country for traceability (if not already there)
            if "_loop_country" not in final.columns:
                final = final.assign(_loop_country=c)

            results.append(final)

        except Exception as e:
            print(f"[ERROR] {c}: {e}")
            failed_countries.append(c)

    print("\nDone.\nFailed countries:", failed_countries)

    combined = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    # optional: save
    # combined.to_csv("yields.csv", index=False)
