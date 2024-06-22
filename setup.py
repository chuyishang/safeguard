from setuptools import setup, find_packages
setup(
    name='aihack',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        # list your package dependencies here
        # e.g., 'requests', 'numpy'
    ],
    python_requires='>=3.9',
    # entry_points={
    #     'console_scripts': [
    #         'aihack2024=aihack2024.your_module:main_function',
    #     ],
    # },
)