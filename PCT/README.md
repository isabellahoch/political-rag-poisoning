# Political Compass Test (PCT)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Overview

The Political Compass Test (PCT) is a Python package that allows users to proctor the political compass test algorithmically by allowing LLMs to react to PCT questions with open-ended responses. It classifies responses in agreement levels according to a specified threshold, then directly interacts with the [PCT website](https://www.politicalcompass.org/test) to determine the generator's political position on the compass.

## Features

- Elicits responses to questions for the political compass test
- Classifies generated responses to determine their political position
- Programmatically intearacts with PCT website to take the test and determine results
- Displays PCT results on interactive plot

## Setup

Expects a file path to a directory with `crx/adblock.crx` extension and subdirectories for generated `response`, `score`, `results` files.
