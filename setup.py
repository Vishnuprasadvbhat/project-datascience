from setuptools import find_packages, setup
# Find packages will automatically find all the packages installed in the while 
#  wrapping the whole aplications as a package
from typing import List


Hypen_dot = '-e .'


def get_requirements(file_path: str)-> List[str]:
  """
  this function returns the requiremnts
  """
  requirements= []
  with open(file_path) as file_obj: 
    requirements = file_obj.readlines()
    requirements=[req.replace('\n', '') for req in requirements]

    if Hypen_dot in requirements:
      requirements.remove(Hypen_dot)

  return requirements




setup(
  name= 'project_datascience',
  version = '0.0.1',
  author='Vishnuprasad',
  author_email= 'vishnuprasadvbhat@gmail.com',
  packages= find_packages(),
  install_requires= get_requirements('requirements.txt')

)