# InformationDiffusion
Project: To find an efficient approach to see information flow change over a period of time on twitter data.
Input: Flume files collected from twitter.Give the flume file at open() statement.
Output: Clusters of topics which are hashtags saved as a text file.
Description: A graph with weighted edges to construct a network for information.For this,'Text' and 'full_text'
             columns are parsed from the json.Hashtags and usernames which is 'screenname' are collected from
             the files. Hashtags are considered as nodes for the graph and usernames connecting hashtags as
             weighed edges.The graph is represented as adjacency matrix and spectral clustering method is used
             to model the network. The clusters formed are used to see the information flow change over a period
             of time for the keyword chosen from the data.
