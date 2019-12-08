---
title: "Quantitative Trading Project"
date: 2019-09-01
#tags: [data wrangling, data science, messy data]
header:
  image: "/images/stock.jpg"
#excerpt: "Data Wrangling, Data Science, Messy Data"
mathjax: "true"
---

# Part 1: Getting Financial Data and Storing Data in MySQL


Note this part assumes you have MySQL, Python, Anaconda installed. 

We will use the mysql.connector to connect to MySQL via Python. If you don't have this package installed, you will need to run this line in cmd:

```cmd
pip install mysql-connector-python
```




## Getting S&P historical daily price



## Setting up MySQL connection in Python


















# H1 Heading

## H2 Heading

### H3 Heading

Here's some basic text.

And here's some *italics*

Here's some **bold** text.

What about a [link](https://github.com/dataoptimal)?

Here's a bulleted list:
* First item
+ Second item
- Third item

Here's a numbered list:
1. First
2. Second
3. Third

Python code block:
```python
    import numpy as np

    def test_function(x, y):
      z = np.sum(x,y)
      return z
```

R code block:
```r
library(tidyverse)
df <- read_csv("some_file.csv")
head(df)
```

Here's some inline code `x+y`.

Here's an image:
<img src="{{ site.url }}{{ site.baseurl }}/images/perceptron/linsep.jpg" alt="linearly separable data">

Here's another image using Kramdown:
![alt]({{ site.url }}{{ site.baseurl }}/images/perceptron/linsep.jpg)

Here's some math:

$$z=x+y$$

You can also put it inline $$z=x+y$$
