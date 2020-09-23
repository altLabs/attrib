"""
trace_ancestry.py

Infer lineages from addgene data
"""

import pandas as pd
from collections import deque
import os
'''
PSEUDOCODE
==========
If set of unseen nodes is nonzero: 
    Get an unseen node
    Current group number increments by 1
    Intialize groups_seen with current group number
    Iniatialize the queue with new node
    While still doing BFS on ancestors:
        Get next node off of the queue
        Get this node's ancestors
        For each ancestor:
            If they're in the queue:
                Move on
            If they're not in the queue but are in global_visited i.e. have been visited in another round:
                Get their group number and add it to groups_seen
            Else:
                Add ancestor to the queue
        Add this node to the local_visited list i.e. the list of nodes visited in this round

    Add local_visited list to groups with group_number
    For each local visited node:
        Add to global_visited with the group number
    
    Merge all groups in groups_seen
    For each node in this group, update global_visited with new group number
'''

def trace_lineages(df):
    '''
    Assign nodes to lineages by doing BFS on adjacency list 
    to find connected components.
    
    === Parameters ===
    df: DataFrame -- adjacency list from plasmid to ancestors
    '''
    
    adjacency_list = df.iloc[:,0].to_dict()
    unvisited = df.index.tolist()
    # {group number: list of nodes}
    groups = {}
    # {node: group}
    global_visited = {}
    i = 1
    
    while len(unvisited) > 0:
        group = i
        groups_seen = {group}
        queue = deque([unvisited.pop(0)])
        local_visited = []
        
        # Each iteration of this loop will create a group
        while len(queue) > 0:
            
            node_id = queue.popleft()
            if node_id in adjacency_list.keys():
            
                for child in adjacency_list[node_id]:
                    # Remove strange loops
                    if child in adjacency_list.keys():
                        if node_id in adjacency_list[child]:
                            local_visited += [child]
                            continue
                            
                    if child in queue:
                        1
                    elif child in global_visited.keys():
                        groups_seen.add(global_visited[child])
                    else:
                        queue.append(child)
                
            local_visited += [node_id]
        
        # Add nodes visited in this round to group dictionary
        groups[group] = local_visited
        # Add nodes and their group numbers
        for local_node in local_visited:
            global_visited[local_node] = group
        
        # Groups get merged into group with smallest index
        group_to_join = min(groups_seen)
            
        # Merge groups
        total = []
        for group_seen in groups_seen:
            total += groups.pop(group_seen)
        groups[group_to_join] = total
              
        for node in groups[group_to_join]:
            global_visited[node] = group_to_join
            
        i += 1
        
    return groups, global_visited

if __name__=='__main__':
    
    processed_data_dir_path = '../../../data/lineages/'
    
    ancestors = pd.read_pickle(os.path.join(processed_data_dir_path,'ancestors.pickle'))
    descendents = pd.read_pickle(os.path.join(processed_data_dir_path, 'descendents.pickle'))
    
    anc_groups, anc_global_visited   = trace_lineages(ancestors)
    desc_groups, desc_global_visited = trace_lineages(descendents)
    
    assert len(anc_groups) == len(desc_groups)
    
    print(f'Number of lineages: {len(anc_groups)}')
    print(f'Number of plasmids with lineages: {len(anc_global_visited)}')
    
    
    