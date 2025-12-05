"""
Diagnostic et correction du problème de communautés identiques
"""

import pickle
import networkx as nx
import community as community_louvain
from collections import Counter

def diagnose_communities(graph_path, partition_path=None):
    """Diagnostiquer le problème des communautés"""
    
    print("=" * 70)
    print("DIAGNOSTIC DES COMMUNAUTÉS")
    print("=" * 70)
    
    # 1. Charger le graphe
    print("\n📥 Chargement du graphe...")
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)
    
    print(f"✅ Graphe chargé : {G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes")
    
    # 2. Charger ou calculer la partition
    if partition_path:
        try:
            print(f"\n📥 Chargement de la partition existante...")
            with open(partition_path, 'rb') as f:
                partition = pickle.load(f)
            print(f"✅ Partition chargée")
        except FileNotFoundError:
            print(f"⚠️ Partition non trouvée, calcul en cours...")
            partition = None
    else:
        partition = None
    
    if partition is None:
        print("\n🔄 Calcul de la partition Louvain...")
        partition = community_louvain.best_partition(G)
        print(f"✅ Partition calculée")
    
    # 3. Analyser la distribution des communautés
    print("\n" + "=" * 70)
    print("ANALYSE DE LA DISTRIBUTION")
    print("=" * 70)
    
    community_counts = Counter(partition.values())
    total_communities = len(community_counts)
    
    print(f"\n📊 Nombre total de communautés : {total_communities}")
    print(f"📊 Taille de la plus grande communauté : {max(community_counts.values())}")
    print(f"📊 Taille de la plus petite communauté : {min(community_counts.values())}")
    
    print(f"\n🔝 Top 10 des communautés par taille :")
    for comm_id, count in community_counts.most_common(10):
        percentage = (count / G.number_of_nodes()) * 100
        print(f"   Communauté {comm_id:3d} : {count:5d} nœuds ({percentage:5.2f}%)")
    
    # 4. PROBLÈME DÉTECTÉ : Une communauté domine
    largest_comm = max(community_counts.items(), key=lambda x: x[1])
    largest_percentage = (largest_comm[1] / G.number_of_nodes()) * 100
    
    print("\n" + "=" * 70)
    print("DIAGNOSTIC")
    print("=" * 70)
    
    if largest_percentage > 80:
        print(f"\n⚠️ PROBLÈME CRITIQUE DÉTECTÉ !")
        print(f"   La communauté {largest_comm[0]} contient {largest_percentage:.1f}% des nœuds")
        print(f"   Cela indique probablement un problème :\n")
        print(f"   1. Le graphe est mal structuré (peu d'arêtes)")
        print(f"   2. La partition n'a pas convergé correctement")
        print(f"   3. Le graphe est dominé par une composante connexe géante\n")
    else:
        print(f"\n✅ Distribution acceptable")
        print(f"   La plus grande communauté ne représente que {largest_percentage:.1f}%")
    
    # 5. Vérifier la connectivité
    print("\n" + "=" * 70)
    print("ANALYSE DE CONNECTIVITÉ")
    print("=" * 70)
    
    if nx.is_connected(G):
        print("\n✅ Le graphe est connexe")
    else:
        components = list(nx.connected_components(G))
        print(f"\n⚠️ Le graphe a {len(components)} composantes connexes")
        largest_comp = max(components, key=len)
        print(f"   Plus grande composante : {len(largest_comp)} nœuds ({len(largest_comp)/G.number_of_nodes()*100:.1f}%)")
    
    # 6. Qualité de la modularité
    print("\n" + "=" * 70)
    print("QUALITÉ DE LA PARTITION")
    print("=" * 70)
    
    modularity = community_louvain.modularity(partition, G)
    print(f"\n📊 Modularité : {modularity:.4f}")
    
    if modularity < 0.3:
        print("   ⚠️ Modularité faible : la structure communautaire est peu marquée")
    elif modularity < 0.5:
        print("   ⚙️ Modularité moyenne : structure communautaire modérée")
    else:
        print("   ✅ Modularité élevée : bonne structure communautaire")
    
    return G, partition, community_counts


def recalculate_communities(G, resolution=1.0, random_state=42):
    """Recalculer les communautés avec paramètres ajustés"""
    
    print("\n" + "=" * 70)
    print("RECALCUL DES COMMUNAUTÉS")
    print("=" * 70)
    
    print(f"\n🔄 Calcul avec resolution={resolution}...")
    
    # Louvain avec résolution ajustée
    partition = community_louvain.best_partition(
        G, 
        resolution=resolution,
        random_state=random_state
    )
    
    community_counts = Counter(partition.values())
    modularity = community_louvain.modularity(partition, G)
    
    print(f"✅ Partition recalculée")
    print(f"   📊 Nombre de communautés : {len(community_counts)}")
    print(f"   📊 Modularité : {modularity:.4f}")
    print(f"   📊 Taille moyenne : {sum(community_counts.values())/len(community_counts):.1f} nœuds")
    
    print(f"\n🔝 Top 5 des communautés :")
    for comm_id, count in community_counts.most_common(5):
        percentage = (count / G.number_of_nodes()) * 100
        print(f"   Communauté {comm_id:3d} : {count:5d} nœuds ({percentage:5.2f}%)")
    
    return partition, modularity


def test_different_resolutions(G):
    """Tester différentes résolutions pour trouver la meilleure"""
    
    print("\n" + "=" * 70)
    print("TEST DE DIFFÉRENTES RÉSOLUTIONS")
    print("=" * 70)
    
    resolutions = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    results = []
    
    for res in resolutions:
        partition = community_louvain.best_partition(G, resolution=res)
        modularity = community_louvain.modularity(partition, G)
        num_communities = len(set(partition.values()))
        
        community_counts = Counter(partition.values())
        largest_percentage = (max(community_counts.values()) / G.number_of_nodes()) * 100
        
        results.append({
            'resolution': res,
            'modularity': modularity,
            'num_communities': num_communities,
            'largest_comm_percentage': largest_percentage
        })
        
        print(f"\n📊 Resolution = {res}")
        print(f"   Modularité : {modularity:.4f}")
        print(f"   Communautés : {num_communities}")
        print(f"   Plus grande : {largest_percentage:.1f}%")
    
    # Trouver la meilleure
    best = max(results, key=lambda x: x['modularity'])
    print("\n" + "=" * 70)
    print(f"🏆 MEILLEURE CONFIGURATION : resolution={best['resolution']}")
    print(f"   Modularité : {best['modularity']:.4f}")
    print(f"   Communautés : {best['num_communities']}")
    print("=" * 70)
    
    return best['resolution']


def fix_communities_and_save(graph_path, output_path, resolution=None):
    """Corriger et sauvegarder la nouvelle partition"""
    
    # 1. Charger et diagnostiquer
    G, old_partition, old_counts = diagnose_communities(graph_path)
    
    # 2. Trouver la meilleure résolution si non spécifiée
    if resolution is None:
        resolution = test_different_resolutions(G)
    
    # 3. Recalculer avec la meilleure résolution
    new_partition, modularity = recalculate_communities(G, resolution=resolution)
    
    # 4. Sauvegarder
    print(f"\n💾 Sauvegarde dans {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(new_partition, f)
    print(f"✅ Partition sauvegardée")
    
    # 5. Comparaison avant/après
    print("\n" + "=" * 70)
    print("COMPARAISON AVANT/APRÈS")
    print("=" * 70)
    
    old_mod = community_louvain.modularity(old_partition, G)
    new_counts = Counter(new_partition.values())
    
    print(f"\nAVANT :")
    print(f"   Communautés : {len(old_counts)}")
    print(f"   Modularité : {old_mod:.4f}")
    print(f"   Plus grande : {max(old_counts.values())/G.number_of_nodes()*100:.1f}%")
    
    print(f"\nAPRÈS :")
    print(f"   Communautés : {len(new_counts)}")
    print(f"   Modularité : {modularity:.4f}")
    print(f"   Plus grande : {max(new_counts.values())/G.number_of_nodes()*100:.1f}%")
    
    improvement = ((modularity - old_mod) / old_mod) * 100
    print(f"\n{'✅' if improvement > 0 else '⚠️'} Amélioration : {improvement:+.1f}%")
    
    return new_partition


# ============= UTILISATION =============

if __name__ == "__main__":
    
    graph_path = "../data/graph_complete.pkl"
    partition_path = "../results/communities/louvain_partition.pkl"
    output_path = "../results/communities/louvain_partition_fixed.pkl"
    
    # Option 1 : Diagnostic seul
    print("\n🔍 MODE DIAGNOSTIC\n")
    G, partition, counts = diagnose_communities(graph_path, partition_path)
    
    