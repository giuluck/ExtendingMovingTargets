import os
import shutil

if __name__ == '__main__':
    # remove ./build folder
    shutil.rmtree('./build', ignore_errors=True)

    # remove .rst files in ./source folder (besides index.rst and modules.rrt)
    for filename in os.listdir('./source'):
        if filename.endswith('.rst') and filename not in ['index.rst', 'modules.rst']:
            os.remove(f'./source/{filename}')

    # generate documentation .rst with sphinx-apidoc for each module
    for module in ['moving_targets', 'src', 'test']:
        os.system(f'sphinx-apidoc -e -T -M -d 8 -o .\\source ..\\{module}')

    # make html
    os.system('sphinx-build .\\source .\\build\\html')
