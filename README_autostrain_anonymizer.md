# AutoStrainCap Anonymizer

Windows-friendly app for anonymizing AutoStrainCap text reports.

It removes these rows from each report:

- `Patient Name`
- `Patient ID`
- `Patient Date of Birth`
- `Patient Sex`

It writes anonymized report files to an output folder and saves one mapping row per report in a CSV file. The mapping includes `Study UID`, `Study ID`, `Study Date and Time`, `Export Date and Time`, the removed patient details, the source file, and the anonymized file.

If a report is already listed in the mapping CSV, or an anonymized file with the same name already exists in the output folder, the app reuses that existing entry/file instead of creating another copied file.

## Run From Python

Open the GUI:

```powershell
python autostrain_anonymizer.py
```

Batch mode:

```powershell
python autostrain_anonymizer.py "global_strain_plots" -o "anonymized_reports" -m "anonymized_reports\anonymization_mapping.csv"
```

## Build EXE

```powershell
.\build_autostrain_anonymizer_exe.ps1
```

The executable is written to:

```text
dist\AutoStrainCapAnonymizer.exe
```
