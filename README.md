### Files

To avoid data protection complexities, the original dataset is not stored in it.
- **data_clean**
  - HepTh_abstracts
    - abstract folds(1992-2003) => original dataset
    - extract.ipynb
    - paper_details.txt => collect all paper details from the folders
    - **updated_paper_details.txt** => Clean paper details
    - paper_titles.txt => split from updated_paper_details.txt
    - paper_dates.txt => split from updated_paper_details.txt
  - Cit_HepTh.txt => original dataset
  - **Filtered_HepTh_edges.txt** => dataset that deleted papers that comes from HepPh
  - dataCleaning.ipynb => data cleaning including data filling and error handling
  - HepTh_edges.txt => original dataset after data cleaning including data filling and error handling
  - missing_nodes.txt


- **node_filter**
- nodeFilter.ipynb => apply largest Scc and weighted pagerank algotithm to the dataset
- HepTh_pagerank_results => the results of pageranks, used for node filtering
- edges.txt => network edges after node filtering


- **web_application**
- paper_dates.txt
- paper_details.txt
- edges.txt
- HepTh_pagerank_results.json
- index.py
Run the file using python to open the interactive 3D citation network visualization tool. You can directly type *python index.py* in your terminal. 
Example: 
```
Dash is running on http://127.0.0.1:8050/

 * Serving Flask app 'index'
 * Debug mode: on
 ...
 ```
 Open the link in the Browser, then you can get the tool.


## Environment

**Operating system**: Windows 10, macOS 14.5 
**Language**: Python 3.8 or higher 
**Environment setting**: 

Python environment should include the following libraries:
- Dash: A Python framework for building Web applications.
- NetworkX: A structural, dynamic (graph) algorithm for creating and manipulating complex networks.
- Plotly: A graphics library for creating interactive charts.
- NumPy: Provides support for large-scale numerical computation.
- Community Louvain: A library of algorithms for community detection
- Dateutil: Provides additional support for dates and times, such as parsing dates in different formats.


