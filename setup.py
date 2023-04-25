from setuptools import find_packages, setup

HYPEN_E_DOT = '-e .'
def get_requirements(file):
    with open(file) as f:
        requirements = f.readlines()
        requirements = [r.replace('\n','') for r in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements

setup(
    name='mlproject',
    version='0.0.1',
    packages=find_packages(),
    # install_requires=['pands','numpy']
    install_requires=get_requirements('requirements.txt')
)