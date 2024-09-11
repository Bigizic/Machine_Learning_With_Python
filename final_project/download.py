#!/usr/bin/env python3

from pyodide.http import pyfetch

async def download(url, filename):
    response = await pyfetch(url)
    if response.status == 200:
        with open(filename, "wb") as f:
            f.write(await response.bytes())

path = "https://www.afile.com/jpeg"
await download(path, "Weather_Data.csv")
filename ="Weather_Data.csv"
