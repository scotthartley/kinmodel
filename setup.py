from setuptools import setup, find_packages


def readme():
    with open("README.rst", encoding='utf8') as file:
        return file.read()


exec(open('kinmodel/_version.py').read())

setup(name='kinmodel',
      version=__version__,
      description='Chemical kinetic model fitting and simulation',
      long_description=readme(),
      author='Scott Hartley',
      author_email='scott.hartley@miamioh.edu',
      url='https://hartleygroup.org',
      license='MIT',
      packages=find_packages(),
      include_package_data=True,
      entry_points={
          'console_scripts': [
              'fit_kinetics = kinmodel:fit_kinetics',
              'model_kinetics = kinmodel:model_kinetics',
          ]
      },
      install_requires=[
          'numpy', 'scipy>=1.2.1', 'matplotlib', 'pathos', 'PyYAML'],
      python_requires=">=3.6",
      )
