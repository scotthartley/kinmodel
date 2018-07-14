from setuptools import setup, find_packages

def readme():
    with open("README.rst") as file:
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
      scripts=['bin/fit_kinetics.py'],
      install_requires=['numpy', 'scipy', 'matplotlib'],
      python_requires=">=3.6",
      )
