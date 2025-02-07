"""
Author: Hayley Wragg
Description: Functions for performing pathway enrichment using ENRICHR API
Created: 6th February 2025
"""

import os
import time
import json
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def _type_dict_check(p_dict):
    """
        Check if the input is a dictionary.
        Args:
            p_dict (Any): The input to check.
        Raises:
            TypeError: If the input is not a dictionary.
        """
    if not isinstance(p_dict, dict):
        error_string = f"p_dict should be a dict, not {type(p_dict)}"
        raise TypeError(error_string)

def _path_id_check(p_df):
    """Check if the input DataFrame contains the 'path_id' column.

        Args:
            p_dict (pd.DataFrame): DataFrame to check.

        Raises:
            ValueError: If 'path_id' column is not found in the DataFrame.
        """
    if 'path_id' not in p_df.columns:
        error_string = f"""p_dict must contain 'path_id' column.
            Available columns: {p_df.columns.tolist()}"""
        raise ValueError(error_string)

def _key_in_dict_check(k, p_dict):
    """Check if a key exists in a dictionary.
        Args:
            k (str): The key to check for.
            p_dict (dict): The dictionary to check in.
        Raises:
            KeyError: If the key is not found in the dictionary.
        """
    if k not in p_dict:
        raise KeyError(f"key '{k}' not found in p_dict")

def _validate_dict_trio(path_clust_all_dict, path_clust_or_dict, path_clust_score_dict):
    """Validates the clust dictionaries of pathway dataframes"""
    _type_dict_check(path_clust_or_dict)
    _type_dict_check(path_clust_all_dict)
    _type_dict_check(path_clust_score_dict)
    for k in path_clust_all_dict.keys():
        _path_id_check(path_clust_all_dict[k])
        if 'pval' not in path_clust_all_dict[k].columns:
            error_string = f"""path_clust_all_dict must contain 'pval' column.
                Available columns: {path_clust_all_dict[k].columns.tolist()}"""
            raise ValueError(error_string)
    for k in path_clust_or_dict.keys():
        _path_id_check(path_clust_or_dict[k])
    for k in path_clust_score_dict.keys():
        _path_id_check(path_clust_score_dict[k])

def get_pathway_rows_from_data(data, c_num_lab):
    """
    Extracts and organizes pathway information from the given data.

    Args:
        data (list): A list containing pathway data with the following elements:
            - rank (int, or str): The rank of the pathway.
            - pathway (str): The name of the pathway.
            - pval (float): The p-value associated with the pathway.
            - OR (float): The odds ratio.
            - score (float): The combined score.
            - overlap_genes (list): A list of overlapping genes.
            - adjust_pval (float): The adjusted p-value.
        c_num_lab (int or str): The cluster number label.

    Returns:
        dict: A dictionary containing three dictionaries:
            - "or_row": dict with keys "ClusterNumber", "pathway", "path_id", and "OddsRatio"
            - "score_row": dict with keys "ClusterNumber", "pathway", "path_id", and "CombinedScore"
            - "all_row": dict with keys "ClusterNumber", "pathway", "path_id", "rank", "pval", 
                        "AdjustedPval", "OddsRatio", "CombinedScore", "NOverlap", and "OverlapGenes"
    """
    # Check input types
    if not isinstance(data, list):
        raise TypeError(f"data should be a list, not {type(data)}")
    # Check that data contains the necessary terms
    if len(data) != 7:
        error_string = """data should contain exactly 7 elements: rank, pathway,
            pval, OR, score, overlap_genes, adjust_pval"""
        raise ValueError(error_string)
    if not isinstance(data[1], str):
        raise TypeError(f"data[1] (pathway) should be a str, not {type(data[1])}")
    if not isinstance(data[2], float):
        raise TypeError(f"data[2] (pval) should be a float, not {type(data[2])}")
    if not isinstance(data[3], float):
        raise TypeError(f"data[3] (OR) should be a float, not {type(data[3])}")
    if not isinstance(data[4], float):
        raise TypeError(f"data[4] (score) should be a float, not {type(data[4])}")
    if not isinstance(data[5], list):
        raise TypeError(f"data[5] (overlap_genes) should be a list, not {type(data[5])}")
    if not isinstance(data[6], float):
        raise TypeError(f"data[6] (adjust_pval) should be a float, not {type(data[6])}")
    rank = data[0]
    pathway = data[1]
    path_full = pathway.split(" R-")
    pathway = path_full[0]
    path_id = "R-"+path_full[1]
    pval = data[2]
    o_rat = data[3]
    score = data[4]
    overlap_genes = data[5]
    adjust_pval = data[6]
    all_row = {"ClusterNumber": c_num_lab,
                "pathway": pathway,
                "path_id": path_id,
                "rank": rank,
                "pval": pval, 
                "AdjustedPval": adjust_pval,
                "OddsRatio" : o_rat,
                "CombinedScore": score,
                "NOverlap": len(overlap_genes),
                "OverlapGenes": overlap_genes}
    or_row = {"ClusterNumber": c_num_lab,
            "pathway": pathway,
            "path_id": path_id,
            "OddsRatio" : o_rat}
    score_row = {"ClusterNumber": c_num_lab,
            "pathway": pathway,
            "path_id": path_id,
            "CombinedScore": score}
    out_dict = {"or_row":or_row, "score_row":score_row, "all_row":all_row}
    return out_dict

def fetch_enrichment(req_ses, gene_library, user_id,
                     max_tries = 20, pause = 10):
    """
    Fetches pathway enrichment results from the ENRICHR API.

    Args:
        req_ses (requests.Session): requests session to use for making the API call.
        gene_library (str): gene library to use for enrichment analysis.
        user_id (str or int): user ID obtained from the ENRICHR API after submitting gene list.
        max_tries (int, optional): max no. attempts to fetch the enrichment results. Defaults to 20.
        pause (int, optional): pause (seconds) between retries for 409 response. Defaults to 10.

    Returns:
        requests.Response: The response object containing the enrichment results.
    """
    # Check input types
    if not isinstance(req_ses, requests.Session):
        raise TypeError(f"req_ses should be a requests.Session, not {type(req_ses)}")
    if not isinstance(gene_library, str):
        raise TypeError(f"gene_library should be a str, not {type(gene_library)}")
    if not isinstance(user_id, (str, int)):
        raise TypeError(f"user_id should be a str or int, not {type(user_id)}")
    if not isinstance(max_tries, int):
        raise TypeError(f"max_tries should be an int, not {type(max_tries)}")
    if not isinstance(pause, int):
        raise TypeError(f"pause should be an int, not {type(pause)}")
    enrichr_url = 'https://maayanlab.cloud/Enrichr/enrich'
    query_string = '?userListId=%s&backgroundType=%s'
    for _ in range(max_tries):
        print("Get enrichment")
        response = req_ses.get(enrichr_url + query_string % (user_id, gene_library))
        if response.ok:
            break
        elif '409' in response:
            time.sleep(pause)
        else:
            print(response)
            print("Error fetching enrichment")
            break
    return response

def enrich_clust(gene_set, meth_key, c_num_lab, gene_library, req_ses, max_tries=20):
    """Perform pathway enrichment analysis for a given gene set.

    This function sends a gene list to the Enrichr API, retrieves enrichment results,
    and processes the results to extract relevant pathway information.
    
    Args:
        gene_set (list): List of genes to be analyzed.
        meth_key (str): String representing the method key.
        c_num_lab (str): String representing the cluster number label.
        gene_library (str): Gene library to be used for enrichment analysis.
        req_ses (requests.Session): Requests session object for making HTTP requests.
        max_tries (int, optional): max no. attempts to send gene list to Enrichr. Defaults to 20.
        pause (int, optional): Pause (seconds) between retry attempts. Defaults to 10.
    Returns:
        dict: A dictionary containing the following keys:
            - "all_list" (list): A list of dictionaries with pathway information for all clusters.
            - "OR_list" (list): A list of dictionaries with odds ratio information for all clusters.
            - "score_list" (list): A list of dictionaries with score information for all clusters.
            - "user_list_id" (str): The user list ID returned by Enrichr.
    """
    # Check input types
    if not isinstance(gene_set, list):
        raise TypeError(f"gene_set should be a list, not {type(gene_set)}")
    if not isinstance(meth_key, str):
        raise TypeError(f"meth_key should be a str, not {type(meth_key)}")
    if not isinstance(c_num_lab, str):
        raise TypeError(f"c_num_lab should be a str, not {type(c_num_lab)}")
    if not isinstance(gene_library, str):
        raise TypeError(f"gene_library should be a str, not {type(gene_library)}")
    if not isinstance(req_ses, requests.Session):
        raise TypeError(f"req_ses should be a requests.Session, not {type(req_ses)}")
    if not isinstance(max_tries, int):
        raise TypeError(f"max_tries should be an int, not {type(max_tries)}")
    enrichr_add_url = 'https://maayanlab.cloud/Enrichr/addList'
    genes_str = '\n'.join(gene_set)
    description = meth_key + c_num_lab + ' gene list'
    payload = {
        'list': (None, genes_str),
        'description': (None, description)
    }
    all_list = []
    score_list = []
    or_list = []
    data = []
    user_id = ""
    # Analyse Genes
    print("Analyse Genes")
    for _ in range(max_tries):
        response_add = req_ses.post(enrichr_add_url, files=payload)
        if response_add.ok:
            data_add = json.loads(response_add.text)
            user_id = data_add["userListId"]
            response_enrich = fetch_enrichment(req_ses, gene_library, user_id)
            data = json.loads(response_enrich.text)[gene_library]
            break
        elif '409' in response_add:
            time.sleep(10)
        else:
            print("payload", payload)
            print(response_add)
            print('Error analyzing gene list')
            break
    if len(data)>0:
        print(f"Found {len(data)} pathways for cluster {c_num_lab} on method {meth_key}")
        score_clust_list = []
        all_clust_list = []
        or_clust_list = []
        # Iterate through each pathway extracting the information
        for i,_ in enumerate(data):
            out_rows = get_pathway_rows_from_data(data[i], c_num_lab)
            score_clust_list += [out_rows["score_row"]]
            all_clust_list += [out_rows["all_row"]]
            or_clust_list += [out_rows["OR_row"]]
    else:
        all_clust_list = [{"ClusterNumber": c_num_lab,
                          "pathway": "NoPathway",
                          "path_id": "No_id",
                          "pval": None}]
        or_clust_list = [{"ClusterNumber": c_num_lab,
                          "pathway": "NoPathway",
                          "path_id": "No_id"}]
        score_clust_list = [{"ClusterNumber": c_num_lab,
                          "pathway": "NoPathway",
                          "path_id": "No_id"}]
    score_list += score_clust_list
    all_list += all_clust_list
    or_list += or_clust_list
    out_dict = {"all_list": all_list,
        "OR_list": or_list,
        "score_list": score_list,
        "user_list_id": user_id}
    return out_dict

def enrich_method(meth_key, gene_clust_dict, gene_set_library, req_ses):
    """
    Enriches the given gene clusters using the specified method.

    Parameters:
        meth_key (str): key representing the method to be used for enrichment.
        gene_clust_dict (dict): dict containing gene clusters.
        gene_set_library (str): label for gene library
        req_ses (object): session object for making requests.

    Returns:
        dict: A dictionary containing the enrichment results with the following keys:
            - "user_ids_list" (list): List of user IDs.
            - "or_df" (pd.DataFrame): DataFrame containing odds ratio results.
            - "score_df" (pd.DataFrame): DataFrame containing score results.
            - "all_df" (pd.DataFrame): DataFrame containing all results.
    """
    # Check input types
    if not isinstance(meth_key, str):
        raise TypeError(f"meth_key should be a str, not {type(meth_key)}")
    _type_dict_check(gene_clust_dict)
    if not isinstance(gene_set_library, str):
        raise TypeError(f"gene_set_library should be a str, not {type(gene_set_library)}")
    if not isinstance(req_ses, requests.Session):
        raise TypeError(f"req_ses should be a requests.Session, not {type(req_ses)}")

    # Check that meth_key exists in gene_clust_dict
    _key_in_dict_check(meth_key, gene_clust_dict)
    # --------------------------------------
    # Compute pathway enrichment for cluster
    clust_genes_df = gene_clust_dict[meth_key]
    c_num_labs = clust_genes_df.columns
    or_list = []
    score_list = []
    user_ids_list = []
    all_list = []
    for c_num_lab in c_num_labs:
        gene_set = clust_genes_df.index[clust_genes_df[c_num_lab] == 1].tolist()
        enrich_dict = enrich_clust(gene_set, meth_key, c_num_lab, gene_set_library, req_ses)
        time.sleep(10)
        score_list += enrich_dict["score_list"]
        all_list += enrich_dict["all_list"]
        or_list += enrich_dict["OR_list"]
        user_ids_list = enrich_dict["user_list_id"]
    score_df = pd.DataFrame(score_list)
    all_df = pd.DataFrame(all_list)
    or_df =  pd.DataFrame(or_list)
    out_dict = {"user_ids_list": user_ids_list,
        "or_df": or_df,
        "score_df": score_df,
        "all_df": all_df 
    }
    return out_dict

def full_pathway_enrichment(clust_dict, path_dir, gene_set_library = 'Reactome_2022'):
    """ Perform full pathway enrichment analysis for given clusters.
    
    Parameters:
        clust_dict (dict): Dictionary containing cluster information.
        path_dir (str): Directory path to save or load enrichment results.
        gene_set_library (str): gene library for enrichment analysis. Default is 'Reactome_2022'.
        
    Returns:
        out_dict (dict): A dictionary containing the enrichment results with the following keys:
            - "all_dict" (dict): Dictionary of all pathway enrichment results.
            - "or_dict" (dict): Dictionary of odds ratio pathway enrichment results.
            - "score_dict" (dict): Dictionary of score pathway enrichment results.
    """
    _type_dict_check(clust_dict)
    if not isinstance(path_dir, str):
        raise TypeError(f"path_dir should be a str, not {type(path_dir)}")
    if not isinstance(gene_set_library, str):
        raise TypeError(f"gene_set_library should be a str, not {type(gene_set_library)}")
    path_clust_or_dict = {}
    path_clust_all_dict = {}
    path_clust_score_dict = {}
    s= requests.Session()
    retry = Retry(connect=30, backoff_factor=120)
    adapter = HTTPAdapter(max_retries=retry)
    s.mount('http://', adapter)
    s.mount('https://', adapter)
    for meth_key in clust_dict.keys():
        print(meth_key)
        all_name = path_dir+"/ClustPathEnrichment"+meth_key+".csv"
        score_name = path_dir+"/ClustPathCombinedScore"+meth_key+".csv"
        or_name = path_dir+"/ClustPathsOR"+meth_key+".csv"
        if os.path.exists(all_name) and os.path.exists(score_name) and os.path.exists(or_name):
            path_clust_all_dict[meth_key] = pd.read_csv(all_name, index_col = 0)
            path_clust_or_dict[meth_key] = pd.read_csv(or_name, index_col = 0)
            path_clust_score_dict[meth_key] = pd.read_csv(score_name, index_col = 0)
        else:
            enrich_meth = enrich_method(meth_key, clust_dict, gene_set_library, s)
            enrich_meth["or_df"].to_csv(or_name)
            enrich_meth["score_df"].to_csv(score_name)
            enrich_meth["all_df"].to_csv(all_name)
            path_clust_all_dict[meth_key] = enrich_meth["all_df"]
            path_clust_or_dict[meth_key] = enrich_meth["or_df"]
            path_clust_score_dict[meth_key] = enrich_meth["score_df"]
    out_dict = {"all_dict": path_clust_all_dict,
                "or_dict": path_clust_or_dict,
                "score_dict": path_clust_score_dict}
    return out_dict

def apply_p_filter(meth_key, all_dict, or_dict, score_dict, p_val_orig = 5E-8):
    """
    Apply a p-value filter to pathway clustering results and update the dictionaries accordingly.

    Args:
        meth_key (str or int): The method used for clustering.
        all_dict (dict): Dictionary containing all pathway clustering results.
        or_dict (dict): Dictionary containing odds ratio pathway clustering results.
        score_dict (dict): Dictionary containing score pathway clustering results.
        p_val_orig (float, optional): Original p-value threshold. Default is 5E-8.

    Returns:
        dict: A dictionary containing updated pathway clustering results with keys:
              - "all_dict": Updated path_clust_all_dict
              - "or_dict": Updated path_clust_or_dict
              - "score_dict": Updated path_clust_score_dict
    """
    def _apply_filter(path_all, path_or, path_score, p_val_orig):
        """Applies the p-value filter to the pathway dataframes."""
        n_paths = path_all.shape[0]
        rem_paths = []
        if n_paths > 0:
            p_val_new = p_val_orig / n_paths
            rem_paths = path_all[path_all.pval <= p_val_new].path_id.tolist()
            path_all = path_all[~path_all.path_id.isin(rem_paths)]
            path_score = path_score[~path_score.path_id.isin(rem_paths)]
            path_or = path_or[~path_or.path_id.isin(rem_paths)]
            path_all.reset_index(drop=True, inplace=True)
            path_score.reset_index(drop=True, inplace=True)
            path_or.reset_index(drop=True, inplace=True)
        return path_all, path_or, path_score, rem_paths

    # Input validation
    _validate_dict_trio(all_dict, or_dict, score_dict)
    _key_in_dict_check(meth_key, all_dict)
    _key_in_dict_check(meth_key, or_dict)
    _key_in_dict_check(meth_key, score_dict)

    if not isinstance(p_val_orig, float):
        raise TypeError(f"p_val_orig should be a float, not {type(p_val_orig)}")

    path_all = all_dict[meth_key]
    path_or = or_dict[meth_key]
    path_score = score_dict[meth_key]

    # Apply p-value filter
    path_all, path_or, path_score, rm_pth = _apply_filter(path_all, path_or, path_score, p_val_orig)

    # Update dictionaries
    all_dict[meth_key] = path_all
    or_dict[meth_key] = path_or
    score_dict[meth_key] = path_score

    print("paths removed ", len(rm_pth))

    out_dict = {"all_dict": all_dict,
                "or_dict": or_dict,
                "score_dict": score_dict}
    return out_dict

def remove_paths_children(path_dir, meth_key, path_dict, pathways_relation_df):
    """Removes pathways children of pathways from the path_dict based on pathways_relation_df.

    Args:
        path_dir (str): directory where the resulting CSV file will be saved.
        meth_key (str): key to access the specific pathway scores in path_dict.
        path_dict (dict): dictionary containing pathway scores.
        pathways_relation_df (pd.DataFrame): DataFrame of pathway parent-child relationships.

    Returns:
        dict: The updated path_dict with specified pathways and their children removed.
    """
    # Check inout Types
    if not isinstance(path_dir, str):
        error_string = f"""path_dir should be a string not {type(path_dir)}"""
        raise TypeError(error_string)
    if not isinstance(meth_key, str):
        error_string = f"""meth_key should be a string not {type(meth_key)}"""
        raise TypeError(error_string)
    _type_dict_check(path_dict)
    if not isinstance(pathways_relation_df, pd.DataFrame):
        error_string = f"""pathways_relation_df should be a pd DataFrame
            not {type(pathways_relation_df)}"""
        raise TypeError(error_string)
    # Check the inputs are valid
    if meth_key not in path_dict:
        _key_in_dict_check(meth_key, path_dict)
    # Check that 'path_id' is a column in all DataFrames in path_dict
    for _, df in path_dict.items():
        _path_id_check(df)
    # Check that 'parent' and 'child' are columns in pathways_relation_df
    if 'parent' not in pathways_relation_df.columns or 'child' not in pathways_relation_df.columns:
        raise ValueError("pathways_relation_df must contain 'parent' and 'child' columns.")

    path_df = path_dict[meth_key]
    n_paths = path_df.shape[0]
    if n_paths>0:
        for path in path_df.path_id.unique():
            path_child_rows = pathways_relation_df[pathways_relation_df.child.isin([path])]
            parent_ids = path_child_rows.parent
            if any(parent_ids.isin(path_df.path_id)):
                path_df = path_df[path_df.path_id != path]
                # Remove the children of the child too
                child_ids = path_child_rows.child
                for c_id in child_ids:
                    path_df = path_df[path_df.path_id != c_id]
        # Check pathway id in hierarchy df. If child then check the parent path.
        # If the parent pathway is also in the results then remove the child.
        path_df.reset_index(drop = True, inplace=True)
        n_paths2 = path_df.shape[0]
        path_df.to_csv(path_dir+"/ClustPathCombinedScore"+meth_key+".csv")
        print("paths removed ", n_paths - n_paths2)
    return path_dict
