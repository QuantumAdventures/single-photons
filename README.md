# Code for the Single Photon

This repo includes the code to simulate a tweezed particle controlled by an optimal control policy. Here we are going to:

* Implement Kalman Filter class to estimate certain states of the system needed for control
* Create an Environment class, where a tweezed particle will be simulated


## Installation

Well, now, how can you run this code? Follow the next steps.

1. First clone this repo in your computer. This can be done
    * Via SSH:

    ```bash
    git clone https://github.com/QuantumAdventures/optical-bottle-beam.git
    ```

    * Via HTTPS:

    ```bash
    git clone git@github.com:QuantumAdventures/optical-bottle-beam.git
    ```

    * Or, by downloading the .zip of the repo and unziping in your computer.

2. Install the requirements, this will set the libraries needed with the correct version in your computer. Inside the main folder of the project run the following command:

```bash
pip install -r requirements.txt
```

3. Inside the repo run the following command: 

    ```bash
    python setup.py install
    ```

## Usage

Now, with everything set, the notebooks can be executed. To create a new script or notebook importing some of the classes, just insert the following imports

```python
from single_photon.enviroment import Particle
from single_photon.estimator import KalmanFilter
```


## License

This code is licensed by:

General Public License version 3.0 [GPL-3.0](https://choosealicense.com/licenses/gpl-3.0/)


Copyright (C) 2022  Quantum Adventures: Pontifical University of Rio de Janeiro
research group for optomechanics, quantum optics and quantum information.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

## Contact

#Name - [Linkedin](https://www.linkedin.com/in) [Email](email)
#Name - [Linkedin](https://www.linkedin.com/in) [Email](email)
#Name - [Linkedin](https://www.linkedin.com/in) [Email](email)
#Name - [Linkedin](https://www.linkedin.com/in) [Email](email)


Project Link: [Repository](https://github.com/QuantumAdventures/single-photon)

## References

To be added in the future, but will look something like this

Authors *Title* **Journal of Randomic Random Things**. Year.
[doi:xx.XXXXX/](doi:xx.XXXX/)