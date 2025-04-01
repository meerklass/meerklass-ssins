# papermill runner script to process multiple observation blocks

import papermill as pm
import csv
import argparse

# Load CSV and Parameters
# papermill to execute one block for h an v pol
def run_notebook(index):

    with open('good_block.csv') as csvfile:
        good_block = list(csv.reader(csvfile))
        
        block= good_block[index]
    parameter1 = {"Block": block[0], "pol":'h'}
    #parameter2 = {"Block": block[0], "pol":'v'}


    pm.execute_notebook("MeerKLASS-SSINS_AnalysisNB-Executable.ipynb", f"SSINS_Flags/output_nb_good_block_only/output-notebook-{parameter1['Block']}-{parameter1['pol']}.ipynb", kernel_name =    "katcali", parameters=parameter1)
    #pm.execute_notebook("MeerKLASS-SSINS_AnalysisNB-Executable.ipynb", f"SSINS_Flags/output_nb_good_block_only/output-notebook-{parameter2['Block']}-{parameter2['pol']}.ipynb", kernel_name =    "katcali", parameters=parameter2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='Execute SSINS notebook on a specific block number',
    )
    parser.add_argument('index', metavar='N', type=int,
                        help='index (row) in good_block.csv')
    args = parser.parse_args()
    run_notebook(args.index)
