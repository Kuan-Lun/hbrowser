# HBrowser (hbrowser)

## Usage

Here's a quick example of how to use HBrowser:

```python
from browser import DriverPass, EHDriver, HVDriver


if __name__ == "__main__":
    driverpass = DriverPass(username="username", password="password")

    with EHDriver(**driverpass.getdict()) as driver:
        driver.punchin()

    with HVDriver(**driverpass.getdict()) as driver:
        driver.monstercheck()
```
