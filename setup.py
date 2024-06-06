#running 'pip install -e .' in the top folder looks at this script. 
# It enables one to import various python scripts throughout this project as python modules 

from setuptools import setup, find_packages

# Read requirements.txt and install
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='faccent', version='1.0.1', 
    packages=find_packages(),
    description='A pytorch tool for feature visualizations of various types',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Chris Hamblin',
    author_email='chrishamblin7@gmail.com',
    url='https://github.com/chrishamblin7/faccent',
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    install_requires=requirements
    )



