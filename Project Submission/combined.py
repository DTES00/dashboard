import pandas as pd

def combine_heuristics_normalized(overlap_ranking, cluster_ranking, weight_overlap=0.5, weight_cluster=0.5):
    """
    Combine bounding box and clustering heuristics using normalized scores.

    Parameters:
        overlap_ranking (DataFrame): Rankings with ['Company1', 'Company2', 'Overlap Percentage'].
        cluster_ranking (DataFrame): Rankings with ['Company1', 'Company2', 'Average Distance'].
        weight_overlap (float): Weight for the overlap percentage in the combined score.
        weight_cluster (float): Weight for the clustering score in the combined score.

    Returns:
        DataFrame: Combined ranking with normalized scores for both heuristics.
    """
    # Normalize bounding box overlap percentage to [0, 1]
    overlap_ranking['Normalized Overlap Score'] = overlap_ranking['Overlap Percentage'] / 100

    # Merge the two rankings on Company1 and Company2
    combined = pd.merge(
        overlap_ranking[['Company1', 'Company2', 'Normalized Overlap Score']],
        cluster_ranking[['Company1', 'Company2', 'Clustering Score']],
        on=['Company1', 'Company2'],
        how='inner'
    )

    # Calculate the combined score
    combined['Combined Score'] = (
        weight_overlap * combined['Normalized Overlap Score'] +
        weight_cluster * combined['Clustering Score']
    )

    # Sort by combined score in descending order
    combined.sort_values(by='Combined Score', ascending=False, inplace=True)

    return combined[['Company1', 'Company2', 'Normalized Overlap Score', 'Clustering Score', 'Combined Score']]

def get_best_partnerships(ranked_pairs):
    used_companies = set()
    best_partnerships = []

    for _, row in ranked_pairs.iterrows():
        company1, company2, combined_score = row['Company1'], row['Company2'], row['Combined Score']
        if company1 not in used_companies and company2 not in used_companies:
                best_partnerships.append(row)
                used_companies.add(company1)
                used_companies.add(company2)

    return pd.DataFrame(best_partnerships, columns=ranked_pairs.columns)

