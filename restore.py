import pandas as pd

# === File paths ===
non_eqpart_file = "Non EQPart Items.xlsx"
mpn_file = "List with MPNs.xlsx"
output_file = "Non EQPart Items - MPN Fuzzy Restored.xlsx"
log_file = "MPN Restored Rows Log.xlsx"

# === Load Excel files ===
non_eq_df = pd.read_excel(non_eqpart_file, sheet_name="data")
mpn_df = pd.read_excel(mpn_file, sheet_name="Item Master Import")

# === Normalize columns ===
non_eq_df.columns = [col.strip() for col in non_eq_df.columns]
mpn_df.columns = [col.strip() for col in mpn_df.columns]

# === Create MPN lookup: (Key_Phrase, MPN) ===
mpn_df["Key_Phrase"] = mpn_df["Description"].str.extract(r'^\s*\S+\s+(.*)$')[0]
mpn_lookup = list(zip(mpn_df["Key_Phrase"].dropna(), mpn_df["Item"].astype(str)))

# === Prepare target dataframe ===
non_eq_df["Item (child)"] = non_eq_df["Item (child)"].astype(str)
non_eq_df["Description"] = non_eq_df["Description"].astype(str)

# === Initialize tracking list ===
updated_rows = []

# === Matching function with tracking ===
def find_mpn_and_log(row):
    description = row["Description"]
    if row["Item Group"] == "EQPART":
        return description  # Skip EQPART items

    for phrase, mpn in mpn_lookup:
        if pd.notna(phrase) and phrase[:25].lower() in description.lower():
            new_description = f"{mpn} {description}"
            updated_rows.append({
                "Company": row.get("Company", ""),
                "Item (child)": row["Item (child)"],
                "Description": description,
                "New Description": new_description,
                "Item Type": row.get("Item Type", ""),
                "Item Group": row.get("Item Group", "")
            })
            return new_description
    return description

# === Apply update ===
non_eq_df["Description"] = non_eq_df.apply(find_mpn_and_log, axis=1)

# === Save restored item master ===
non_eq_df.to_excel(output_file, sheet_name="data", index=False)

# === Save log of updated rows ===
if updated_rows:
    log_df = pd.DataFrame(updated_rows)
    log_df.to_excel(log_file, index=False)
    print(f"✅ Restored file saved as: {output_file}")
    print(f"📄 Log of updated rows saved as: {log_file}")
else:
    print("⚠️ No rows were updated. Check input data.")

