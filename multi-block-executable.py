# papermill runner script to process multiple observation blocks

import papermill as pm

# Load Parameters
parameters = [{"Block": '1630519596', "pol": 'h'}, {"Block": '1630519596',"pol": 'v'}]


for params in parameters:
    pm.execute_notebook("MeerKLASS-SSINS_AnalysisNB-Executable.ipynb", f"SSINS_Flags/output-notebook-{params['Block']}-{params['pol']}.ipynb", kernel_name =    "katcali", parameters=params)

