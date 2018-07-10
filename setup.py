from setuptools import setup

def readme():
    with open("README.rst") as file:
        return file.read()

setup(name='kinmodel',
      version='0.9',
      description='Chemical kinetic model fitting and simulation',
      long_description=readme(),
      author='Scott Hartley',
      author_email='scott.hartley@miamioh.edu',
      url='https://hartleygroup.org',
      license='MIT',
      packages=['kinmodel'],
      scripts=['bin/fit_kinetics.py'],
      install_requires=['numpy', 'scipy', 'matplotlib'],
      python_requires=">=3.6",
      )
