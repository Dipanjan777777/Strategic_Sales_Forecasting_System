from setuptools import setup, find_packages

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> list[str]:
    '''This function will return the list of requirements'''
    requirements = []
    with open(file_path, 'r') as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n', '') for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements

setup(
   name='Strategic-Sales-Forecasting',
   version='0.0.1',
   description='A useful module',
   author='Dipanjan Santra',
   author_email='dipanjansantra2019@gmail.com',
   packages=find_packages(),  # Automatically find packages
   install_requires=get_requirements('requirements.txt'),  # Parse dependencies from requirements.txt
)