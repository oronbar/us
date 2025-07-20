@echo off
REM Usage: run_plot_xml.bat "path\to\your.xml"

"D:\Code\.venv\Scripts\python.exe" "%~dp0plot_xml.py" --xml_file "%1" --no_show