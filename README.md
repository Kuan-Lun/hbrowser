# HBrowser (hbrowser)

## Usage

Here's a quick example of how to use HBrowser:

```python
from hbrowser import DriverPass, EHDriver


if __name__ == "__main__":
    driverpass = DriverPass(username="username", password="password")

    with EHDriver(**driverpass.getdict()) as driver:
        driver.punchin()
```

Here's a quick example of how to use HVBrowser:

```python
from hbrowser import DriverPass
from hvbrowser import HVDriver


if __name__ == "__main__":
    driverpass = DriverPass(username="username", password="password")

    with HVDriver(**driverpass.getdict()) as driver:
        driver.monstercheck()
```
