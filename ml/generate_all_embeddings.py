import torch
import networkx as nx
import numpy as np
from torch_geometric.nn import SAGEConv
import torch.nn as nn
import torch.nn.functional as F
import pickle
from neo4j import GraphDatabase
import os

# ========== CLASSES DU MODÈLE ==========
class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class EdgePredictor(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.lin = nn.Linear(embedding_dim * 2, 1)
    
    def forward(self, z, edge_index):
        src = z[edge_index[0]]
        dst = z[edge_index[1]]
        out = torch.cat([src, dst], dim=1)
        return torch.sigmoid(self.lin(out)).squeeze()

# ========== FONCTIONS D'AIDE ==========
def load_complete_graph_from_neo4j():
    """Charger TOUS les auteurs depuis Neo4j"""
    
    # Tes credentials Neo4j Aura
    NEO4J_URI = "neo4j+s://c0d3b4ca.databases.neo4j.io"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "a7Pxd2CxrqsYpXhWsmb7kFpTX9Wnw8ofB-2WNkzfUZk"
    
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD)
    )
    
    G = nx.Graph()
    
    print("📥 Connexion à Neo4j pour charger le graphe complet...")
    
    try:
        with driver.session() as session:
            # ✅ Charger tous les auteurs
            print("📥 Chargement des auteurs...")
            result = session.run("""
                MATCH (a:authors)
                WHERE a.authorId IS NOT NULL
                RETURN a.authorId AS id, a.name AS name
            """)
            
            author_count = 0
            for record in result:
                author_id = record["id"]
                if author_id:
                    G.add_node(author_id, name=record["name"])
                    author_count += 1
            
            print(f"✅ {author_count} auteurs chargés")
            
            # ✅ Charger toutes les collaborations
            print("📥 Chargement des collaborations...")
            result = session.run("""
                MATCH (a1:authors)-[:AUTHORED]->(p:papers)<-[:AUTHORED]-(a2:authors)
                WHERE a1.authorId IS NOT NULL AND a2.authorId IS NOT NULL 
                    AND a1.authorId < a2.authorId
                RETURN a1.authorId AS src, a2.authorId AS dst
            """)
            
            edge_count = 0
            for record in result:
                src = record["src"]
                dst = record["dst"]
                if src and dst and G.has_node(src) and G.has_node(dst):
                    G.add_edge(src, dst)
                    edge_count += 1
            
            print(f"✅ {edge_count} collaborations chargées")
    
    except Exception as e:
        print(f"❌ Erreur lors du chargement depuis Neo4j: {e}")
        raise e
    
    finally:
        driver.close()
    
    # Statistiques
    isolated = list(nx.isolates(G))
    print(f"ℹ️  {len(isolated)} auteurs isolés (sans collaborations)")
    print(f"📊 Graphe: {G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes")
    
    return G

def compute_louvain_communities(G):
    """Calculer les communautés avec l'algorithme de Louvain"""
    print("🔍 Calcul des communautés avec Louvain...")
    try:
        import community as community_louvain
        partition = community_louvain.best_partition(G)
        print(f"✅ {len(set(partition.values()))} communautés détectées")
        return partition
    except Exception as e:
        print(f"⚠️  Erreur calcul communautés: {e}")
        # Partition par défaut (tous dans la même communauté)
        return {node: 0 for node in G.nodes()}

def compute_global_metrics(G, partition):
    """Calculer toutes les métriques pour le graphe"""
    print("📊 Calcul des métriques globales...")
    
    metrics_cache = {}
    
    # Degré
    print("  → Degré")
    degrees = dict(G.degree())
    
    # Centralité de degré
    print("  → Centralité de degré")
    degree_cent = nx.degree_centrality(G)
    
    # PageRank (version simplifiée pour grand graphe)
    print("  → PageRank")
    try:
        if G.number_of_nodes() > 10000:
            pagerank = nx.pagerank(G, max_iter=20)
        else:
            pagerank = nx.pagerank(G, max_iter=50)
    except:
        # Fallback simple
        pagerank = {node: 1.0/G.number_of_nodes() for node in G.nodes()}
    
    # Clustering coefficient
    print("  → Coefficient de clustering")
    try:
        clustering = nx.clustering(G)
    except:
        clustering = {node: 0.0 for node in G.nodes()}
    
    # Stocker dans le cache
    for node in G.nodes():
        metrics_cache[node] = {
            'degree': degrees.get(node, 0),
            'degree_centrality': degree_cent.get(node, 0),
            'pagerank': pagerank.get(node, 0),
            'clustering_coefficient': clustering.get(node, 0),
            'community': partition.get(node, -1)
        }
    
    print(f"✅ {len(metrics_cache)} métriques calculées")
    return metrics_cache

# ========== FONCTION PRINCIPALE ==========
def main():
    print("=" * 60)
    print("🎯 GÉNÉRATION DES EMBEDDINGS POUR TOUS LES AUTEURS")
    print("=" * 60)
    
    # 1. Charger le graphe complet depuis Neo4j
    print("\n1️⃣ Chargement du graphe depuis Neo4j...")
    G_full = load_complete_graph_from_neo4j()
    
    # Sauvegarder le graphe pour référence
    with open("data/graph_complete.pkl", "wb") as f:
        pickle.dump(G_full, f)
    print("💾 Graphe complet sauvegardé dans data/graph_complete.pkl")
    
    # 2. Créer node_to_idx pour tous les auteurs
    print("\n2️⃣ Création des mappings...")
    node_list = list(G_full.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    idx_to_node = {v: k for k, v in node_to_idx.items()}
    
    print(f"✅ {len(node_to_idx)} auteurs dans le mapping")
    
    # 3. Calculer les communautés
    print("\n3️⃣ Calcul des communautés...")
    partition = compute_louvain_communities(G_full)
    
    # 4. Calculer les métriques
    print("\n4️⃣ Calcul des métriques...")
    metrics_cache = compute_global_metrics(G_full, partition)
    
    # Sauvegarder le cache de métriques
    with open("results/metrics_cache_complete.pkl", "wb") as f:
        pickle.dump(metrics_cache, f)
    print("💾 Cache de métriques sauvegardé")
    
    # 5. Charger le modèle pré-entraîné
    print("\n5️⃣ Chargement du modèle pré-entraîné...")
    
    # Vérifier si le modèle existe
    model_path = "models/link_prediction_model.pkl"
    if not os.path.exists(model_path):
        print("❌ Modèle non trouvé. Exécute d'abord l'entraînement.")
        return
    
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    
    # 6. Générer les features
    print("\n6️⃣ Génération des features...")
    
    # Features simplifiées: [degree, pagerank, clustering]
    features_list = []
    for node in node_list:
        metrics = metrics_cache.get(node, {})
        features = [
            metrics.get('degree', 0),
            metrics.get('pagerank', 0),
            metrics.get('clustering_coefficient', 0)
        ]
        features_list.append(features)
    
    x = torch.tensor(features_list, dtype=torch.float)
    print(f"✅ Features: {x.shape}")
    
    # 7. Créer edge_index
    print("\n7️⃣ Création des arêtes...")
    edges = []
    for u, v in G_full.edges():
        if u in node_to_idx and v in node_to_idx:
            edges.append((node_to_idx[u], node_to_idx[v]))
    
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        print(f"✅ Edge index: {edge_index.shape}")
    else:
        edge_index = torch.tensor([[], []], dtype=torch.long)
        print("⚠️  Aucune arête trouvée")
    
    # 8. Initialiser l'encodeur
    print("\n8️⃣ Initialisation du modèle...")
    embedding_dim = 64
    encoder = GraphSAGE(3, 128, embedding_dim)
    
    # Charger les poids si disponibles
    if 'model_state' in checkpoint:
        encoder.load_state_dict(checkpoint['model_state']['encoder'])
        print("✅ Poids de l'encodeur chargés")
    else:
        print("⚠️  Poids non trouvés, initialisation aléatoire")
    
    encoder.eval()
    
    # 9. Générer les embeddings
    print("\n9️⃣ Génération des embeddings...")
    with torch.no_grad():
        if edge_index.shape[1] > 0:
            embeddings = encoder(x, edge_index)
        else:
            # Si pas d'arêtes, utiliser features comme embeddings
            embeddings = x
    
    print(f"✅ Embeddings: {embeddings.shape}")
    
    # 10. Créer le modèle étendu
    print("\n🔟 Création du modèle étendu...")
    extended_model = {
        'embeddings': embeddings,
        'node_to_idx': node_to_idx,
        'idx_to_node': idx_to_node,
        'G_full': G_full,
        'partition': partition,
        'metrics_cache': metrics_cache,
        'model_state': checkpoint.get('model_state', {})
    }
    
    # 11. Sauvegarder le modèle étendu
    output_path = "models/link_prediction_model_extended.pkl"
    torch.save(extended_model, output_path)
    
    print("=" * 60)
    print("✅ MODÈLE ÉTENDU GÉNÉRÉ AVEC SUCCÈS")
    print("=" * 60)
    print(f"📁 Fichier: {output_path}")
    print(f"👥 Auteurs: {len(node_to_idx)}")
    print(f"🤝 Arêtes: {len(edges)}")
    print(f"🏢 Communautés: {len(set(partition.values()))}")
    print(f"📊 Embeddings: {embeddings.shape}")
    print("=" * 60)

if __name__ == "__main__":
    main()