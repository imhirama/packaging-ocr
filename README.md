# Quality.AI

An exploratory computer vision project conducted for a global retail client during my time with Fellowship.AI. The company, which brings 2,000+ consumer packaged goods to market each year, wished to automate the QA process for their product packaging. The full project used computer vision tools and techniques to automate over 90% of the task, reducing processing time from hours or days to under ten minutes per product. 

The files here consistute a demo of the exploratory work, conducting scene text detection and OCR using the Google Vision API, and quality control using functions based on the Levenshtein distance package.

## [**packaging_ocr_analysis.ipynb**](packaging_ocr_analysis.ipynb)
Notebook demonstrating the text detection, OCR, and anlysis process. Uses functions from the following two scripts.

## [**product_packaging_ocr.py**](product_packaging_ocr.py)
The core functions for text detection, interpretation, and comparison to groundtruth.

## [**product_packaging_ocr_iterative.py**](product_packaging_ocr_iterative.py)
Calls core functions multiple times for improved detection accuracy.
