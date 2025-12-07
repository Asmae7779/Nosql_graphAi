from fastapi import FastAPI, HTTPException, Path, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict
import torch
import yaml
import pickle
import numpy as np
import networkx as nx
from contextlib import asynccontextmanager
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from neo4j import GraphDatabase
import os

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

class Neo4jClient:
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI", "neo4j+s://c0d3b4ca.databases.neo4j.io")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "a7Pxd2CxrqsYpXhWsmb7kFpTX9Wnw8ofB-2WNkzfUZk")
        self.driver = None
    
    def connect(self):
        self.driver = GraphDatabase.driver(
            self.uri, 
            auth=(self.user, self.password)
        )
        return self
    
    def test_connection(self):
        """Tester la connexion et vérifier un auteur spécifique"""
        try:
            self.connect()
            with self.driver.session() as session:
                # Test 1: Vérifier la connexion
                result = session.run("RETURN 'Connected' as status")
                print("✅ Connexion Neo4j établie")
            
                # Test 2: Vérifier l'auteur 3308557
                query = """
                MATCH (a:authors)
                WHERE toString(a.authorId) = '3308557'
                RETURN a.authorId as id, a.name as name
                LIMIT 1
                """
                result = session.run(query)
                record = result.single()
            
                if record:
                    print(f"✅ Auteur 3308557 trouvé: {record['name']}")
                    return True
                else:
                    print("❌ Auteur 3308557 NON TROUVÉ")
                    return False
                
        except Exception as e:
            print(f"❌ Erreur connexion Neo4j: {e}")
            return False
    
    def get_author_names(self, author_ids):
        """Récupère les noms des auteurs depuis Neo4j - VERSION CORRIGÉE POUR INTEGER"""
        if not self.driver:
            self.connect()

        if not author_ids:
            return {}

        print(f"🔍 [NEO4J INTEGER] Récupération des noms pour {len(author_ids)} auteurs...")

        try:
            # Convertir tous les IDs en integers si possible
            processed_ids = []
            for aid in author_ids:
                try:
                    # Essayer de convertir en int
                    aid_int = int(aid)
                    processed_ids.append(aid_int)
                except:
                    # Si pas convertible, garder tel quel
                    processed_ids.append(aid)
            
            # DIVISER EN 2 STRATÉGIES pour gérer les integers correctement
            all_names = {}
            
            # STRATÉGIE 1: Chercher avec WHERE a.authorId IN $authorIds (pour integers)
            batch_size = 500
            for i in range(0, len(processed_ids), batch_size):
                batch = processed_ids[i:i+batch_size]
                
                # Vérifier si le batch contient des integers
                int_batch = [aid for aid in batch if isinstance(aid, int) or (isinstance(aid, str) and aid.isdigit())]
                
                if int_batch:
                    # Convertir les strings en int pour la requête
                    query_ids = []
                    for aid in int_batch:
                        try:
                            query_ids.append(int(aid))
                        except:
                            query_ids.append(aid)
                    
                    query = """
                    UNWIND $authorIds AS authorId
                    MATCH (a:authors)
                    WHERE a.authorId = authorId
                    RETURN a.authorId as id, a.name as name
                    """
                    
                    try:
                        with self.driver.session() as session:
                            result = session.run(query, {"authorIds": query_ids})
                            
                            for record in result:
                                author_id = record["id"]
                                name = record["name"]
                                # Stocker avec la clé comme string pour uniformité
                                all_names[str(author_id)] = name
                    except Exception as e:
                        print(f"  ⚠️ Erreur lot {i//batch_size + 1} (int): {e}")
        
            # STRATÉGIE 2: Pour ceux qui n'ont pas été trouvés, chercher un par un
            missing_ids = [str(aid) for aid in processed_ids if str(aid) not in all_names]
        
            if missing_ids:
                print(f"  🔍 Recherche des {len(missing_ids)} IDs manquants...")
            
                query = """
                UNWIND $authorIds AS authorId
                MATCH (a:authors)
                WHERE toString(a.authorId) = authorId
                RETURN toString(a.authorId) as id, a.name as name
                """
            
                try:
                    with self.driver.session() as session:
                        result = session.run(query, {"authorIds": missing_ids})
                        
                        for record in result:
                            author_id = record["id"]
                            name = record["name"]
                            all_names[author_id] = name
                except Exception as e:
                    print(f"  ⚠️ Erreur recherche missing IDs: {e}")
        
            # Compléter avec des noms génériques pour ceux toujours manquants
            for aid in processed_ids:
                aid_str = str(aid)
                if aid_str not in all_names:
                    # D'abord essayer de chercher dans le graphe ML
                    try:
                        if isinstance(aid, int):
                            node_id = aid
                        else:
                            try:
                                node_id = int(aid)
                            except:
                                node_id = aid
                        
                        if hasattr(analytics, 'G_full') and node_id in analytics.G_full.nodes():
                            graph_name = analytics.G_full.nodes[node_id].get('name')
                            if graph_name:
                                all_names[aid_str] = graph_name
                                continue
                    except:
                        pass
                    
                    # Sinon nom générique
                    all_names[aid_str] = f"Author {aid}"

            print(f"✅ [NEO4J] {len(all_names)} noms récupérés")
            return all_names

        except Exception as e:
            print(f"❌ Erreur générale get_author_names: {e}")
                # Fallback basique
            return {str(aid): f"Author {aid}" for aid in author_ids}

    def _get_names_from_graph(self, author_ids):
        """Récupère les noms depuis le graphe ML"""
        names = {}
        if not hasattr(analytics, 'G_full'):
            # Fallback basique
            for aid in author_ids:
                names[str(aid)] = f"Author {aid}"
            return names
    
        for aid in author_ids:
            try:
                # Essayer de convertir en int
                try:
                    aid_int = int(aid)
                    node_id = aid_int
                except:
                    node_id = aid
                
                if node_id in analytics.G_full.nodes():
                    graph_name = analytics.G_full.nodes[node_id].get('name')
                    if graph_name:
                        names[str(aid)] = graph_name
                    else:
                        names[str(aid)] = f"Author {aid}"
                else:
                    names[str(aid)] = f"Author {aid}"
            except:
                names[str(aid)] = f"Author {aid}"
    
        print(f"📊 [GRAPH FALLBACK] {len(names)} noms depuis le graphe")
        return names
    
    def _get_author_names_alternative(self, author_ids, session):
        """Alternative: récupérer par lots avec différentes stratégies"""
        all_names = {}
        
        # Parcourir par lots de 100
        batch_size = 100
        for i in range(0, len(author_ids), batch_size):
            batch = author_ids[i:i+batch_size]
            
            # STRATÉGIE A: Essayer avec IDs comme strings
            try:
                string_ids = [str(aid) for aid in batch]
                query = """
                    MATCH (a:authors)
                    WHERE toString(a.authorId) IN $authorIds
                    RETURN toString(a.authorId) AS id, a.name AS name
                """
                result = session.run(query, {"authorIds": string_ids})
                
                for record in result:
                    author_id = record["id"]
                    name = record["name"]
                    if author_id and name:
                        all_names[author_id] = name
                
            except Exception as e:
                print(f"⚠️  [NEO4J] Batch {i//batch_size}: stratégie string échouée: {e}")
            
            # STRATÉGIE B: Essayer ID par ID (plus lent mais plus sûr)
            if len(all_names) < len(batch) * 0.3:  # Si moins de 30% trouvés
                for aid in batch:
                    try:
                        # Essayer de convertir en int
                        try:
                            numeric_id = int(aid)
                            param = numeric_id
                        except:
                            param = aid
                        
                        query = """
                            MATCH (a:authors {authorId: $authorId})
                            RETURN a.authorId AS id, a.name AS name
                        """
                        result = session.run(query, {"authorId": param})
                        
                        for record in result:
                            author_id = record["id"]
                            name = record["name"]
                            if author_id and name:
                                key = str(author_id) if isinstance(author_id, int) else str(author_id)
                                all_names[key] = name
                                break
                                
                    except Exception as e:
                        # Ajouter un nom générique
                        all_names[str(aid)] = f"Author {aid}"
            
            print(f"📦 [NEO4J] Lot {i//batch_size}: {len(all_names) - i} noms accumulés")
        
        print(f"✅ [NEO4J] Total noms récupérés: {len(all_names)}/{len(author_ids)}")
        
        # Compléter avec des noms génériques pour ceux manquants
        for aid in author_ids:
            aid_str = str(aid)
            if aid_str not in all_names:
                all_names[aid_str] = f"Author {aid}"
        
        return all_names
    
    def close(self):
        if self.driver:
            self.driver.close()

# Créer une instance globale
neo4j_client = Neo4jClient()

# ============= GESTIONNAIRE DE MODÈLE =============
class ResearcherAnalytics:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = None
        self.predictor = None
        self.embeddings = None
        self.node_to_idx = None
        self.idx_to_node = None
        self.G_nx = None
        self.G_full = None
        self.partition = None
        self.metrics_cache = {}
        
    def load_model(self, model_path: str, config_path: str):
        """Charger le modèle étendu"""
        print(f"🔄 Chargement du modèle étendu depuis {model_path}...")
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        self.embeddings = checkpoint['embeddings'].to(self.device)
        self.node_to_idx = checkpoint['node_to_idx']
        self.idx_to_node = {v: k for k, v in self.node_to_idx.items()}
        self.G_full = checkpoint['G_full']
        
        # Initialiser G_nx
        self.G_nx = nx.Graph()
        for node in self.G_full.nodes():
            self.G_nx.add_node(node)
        for u, v in self.G_full.edges():
            self.G_nx.add_edge(u, v)
        
        print(f"✅ Modèle étendu chargé: {len(self.node_to_idx)} auteurs")
        
        # Calculer les communautés
        if 'partition' in checkpoint:
            self.partition = checkpoint['partition']
        else:
            print("🔍 Calcul des communautés...")
            import community as community_louvain
            self.partition = community_louvain.best_partition(self.G_nx)
        
        print(f"✅ {len(set(self.partition.values()))} communautés détectées")
        
        # Initialiser encoder et predictor
        embedding_dim = self.embeddings.shape[1]
        self.encoder = GraphSAGE(3, 128, embedding_dim).to(self.device)
        self.predictor = EdgePredictor(embedding_dim).to(self.device)
        
        if 'model_state' in checkpoint:
            self.encoder.load_state_dict(checkpoint['model_state']['encoder'])
            self.predictor.load_state_dict(checkpoint['model_state']['predictor'])
        
        self.encoder.eval()
        self.predictor.eval()
        
        # Pré-calculer les métriques
        self._compute_global_metrics()
    
    def _compute_global_metrics(self):
        """Calculer métriques pour tous les nœuds (avec cache sur disque)"""
        import os
        
        cache_file = "../results/metrics_cache.pkl"
        
        if os.path.exists(cache_file):
            print(f"📦 Chargement des métriques depuis cache...")
            try:
                with open(cache_file, 'rb') as f:
                    self.metrics_cache = pickle.load(f)
                print(f"✅ {len(self.metrics_cache)} métriques chargées depuis cache")
                return
            except:
                print("⚠️ Cache corrompu, recalcul...")
        
        # Si pas de cache, calculer
        print("⏳ Calcul des métriques (peut prendre 1-2 minutes)...")
        self.metrics_cache = {}
        
        print("  → Degree centrality")
        degree_cent = nx.degree_centrality(self.G_nx)
        
        # Betweenness - échantillon uniquement si grand graphe
        if self.G_nx.number_of_nodes() > 2000:
            print("  → Betweenness (échantillon)")
            k = min(500, self.G_nx.number_of_nodes())
            between_cent = nx.betweenness_centrality(self.G_nx, k=k)
        else:
            print("  → Betweenness (complet)")
            between_cent = nx.betweenness_centrality(self.G_nx)
        
        print("  → Closeness")
        closeness_cent = nx.closeness_centrality(self.G_nx)
        
        print("  → PageRank")
        pagerank = nx.pagerank(self.G_nx, max_iter=50)
        
        print("  → Clustering")
        clustering = nx.clustering(self.G_nx)
        
        # Stocker dans cache mémoire
        for node in self.G_nx.nodes():
            self.metrics_cache[node] = {
                'degree': self.G_nx.degree(node),
                'degree_centrality': degree_cent[node],
                'betweenness_centrality': between_cent.get(node, 0),
                'closeness_centrality': closeness_cent[node],
                'pagerank': pagerank[node],
                'clustering_coefficient': clustering[node],
                'community': self.partition.get(node, -1)
            }
        
        # Sauvegarder sur disque
        print(f"💾 Sauvegarde du cache dans {cache_file}...")
        with open(cache_file, 'wb') as f:
            pickle.dump(self.metrics_cache, f)
        
        print(f"✅ {len(self.metrics_cache)} métriques calculées et sauvegardées")
    
    def _fallback_for_isolated(self, top_k):
        """Fallback pour auteurs isolés : proposer top auteurs influents du graphe."""
        # Top par degree
        sorted_degree = sorted(
            self.G_full.degree(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        # Top par PageRank
        sorted_pr = sorted(
            self.metrics_cache.items(),
            key=lambda x: x[1]['pagerank'],
            reverse=True
        )[:top_k]
        
        # Combiner résultats
        combined = {}
        
        for node, deg in sorted_degree:
            combined[node] = {
                "degree": deg,
                "pagerank": self.metrics_cache.get(node, {}).get("pagerank", 0),
                "reason": "Top-degree global (fallback)"
            }
        
        for node, stats in sorted_pr:
            if node not in combined:
                combined[node] = {
                    "degree": self.G_full.degree(node),
                    "pagerank": stats["pagerank"],
                    "reason": "Top PageRank global (fallback)"
                }
        
        # Retourner top_k uniques
        top_nodes = list(combined.keys())[:top_k]
        
        return [
            {
                "researcher_id": node,
                "degree": combined[node]["degree"],
                "pagerank": combined[node]["pagerank"],
                "reason": combined[node]["reason"]
            }
            for node in top_nodes
        ]
    
    def is_author_in_model(self, author_id):
        """Vérifie si un auteur est dans le modèle (avec différentes représentations)"""
        # Essayer différentes représentations
        representations = [author_id]
        
        # Essayer comme int
        try:
            representations.append(int(author_id))
        except:
            pass
    
        # Essayer comme string
        representations.append(str(author_id))
    
        # Chercher dans toutes les représentations
        for rep in representations:
            if rep in self.node_to_idx:
                return True, rep
    
        return False, author_id
    
    def get_recommendations(self, researcher_id: str, top_k: int = 10, 
                            strategy: str = "ml_only", filter_hubs: bool = True,
                            hub_threshold_percentile: int = 95):
        """
        Version corrigée qui utilise is_author_in_model
        """
        print(f"🎯 [GET_RECOMMENDATIONS] Début pour ID: {researcher_id}, type: {type(researcher_id)}")
        
        # Vérifier si l'auteur est dans le modèle
        in_model, actual_id = self.is_author_in_model(researcher_id)
        
        print(f"🔍 [GET_RECOMMENDATIONS] in_model: {in_model}, actual_id: {actual_id}, type: {type(actual_id)}")
        
        if not in_model:
            # Fallback: chercher dans le graphe complet
            print(f"⚠️ Auteur {researcher_id} non trouvé dans le modèle ML, fallback au graphe complet")
            
            if actual_id in self.G_full:
                # Auteur trouvé dans le graphe complet mais pas dans le modèle ML
                # Retourner des recommandations basiques (collaborateurs des collaborateurs)
                neighbors = list(self.G_full.neighbors(actual_id))
                
                if neighbors:
                    # Recommander les collaborateurs des collaborateurs
                    recommendations = []
                    for neighbor in neighbors[:10]:  # Prendre max 10 voisins directs
                        for second_neighbor in self.G_full.neighbors(neighbor):
                            if (second_neighbor != actual_id and 
                                second_neighbor not in neighbors and
                                second_neighbor in self.node_to_idx):  # S'assurer qu'il est dans le modèle
                                
                                # Calculer un score basique
                                degree = self.G_full.degree(second_neighbor)
                                common_neighbors = len(set(self.G_full.neighbors(actual_id)) & 
                                                      set(self.G_full.neighbors(second_neighbor)))
                            
                                score = 0.3 + (min(common_neighbors, 5) * 0.1)  # Score entre 0.3 et 0.8
                            
                                recommendations.append({
                                    "researcher_id": second_neighbor,
                                    "collaboration_score": round(score, 4),
                                    "degree": degree,
                                    "reason": f"Collaborateur de {neighbor} (votre collaborateur)"
                                })
                
                    # Trier et limiter
                    recommendations.sort(key=lambda x: x["collaboration_score"], reverse=True)
                    return recommendations[:top_k]
        
            # Si pas de voisins ou pas trouvé du tout
            return self._fallback_for_isolated(top_k)
        
        # Si on arrive ici, l'auteur est dans le modèle
        researcher_id = actual_id  # Utiliser l'ID normalisé
    
        print(f"✅ [GET_RECOMMENDATIONS] Auteur dans le modèle avec ID: {researcher_id}")
    
        # Vérifier que l'auteur est bien dans G_nx
        if researcher_id not in self.G_nx or self.G_nx.degree(researcher_id) == 0:
            print(f"⚠️ [GET_RECOMMENDATIONS] Auteur isolé ou pas dans G_nx, fallback")
            return self._fallback_for_isolated(top_k)
    
        src_idx = self.node_to_idx[researcher_id]
        src_neighbors = set(self.G_nx.neighbors(researcher_id))
        src_community = self.partition.get(researcher_id, -1)
        src_degree = self.G_nx.degree(researcher_id)
    
        print(f"📊 [GET_RECOMMENDATIONS] Auteur: degré={src_degree}, communauté={src_community}, voisins={len(src_neighbors)}")
    
        # ========== 1. FILTRAGE STRICT DES HUBS ==========
        all_degrees = [self.G_nx.degree(n) for n in self.G_nx.nodes()]
    
        # Seuils percentiles
        p_threshold = np.percentile(all_degrees, hub_threshold_percentile)
        p90 = np.percentile(all_degrees, 90)
        median_deg = np.median(all_degrees)
    
        def should_exclude(node):
            deg = self.G_nx.degree(node)
        
            # Exclure selon le seuil paramétrable
            if deg > p_threshold:
                return True
        
            # Exclure top 10% si l'auteur source est dans la moitié inférieure
            if src_degree < median_deg and deg > p90:
                return True
            
            return False
    
        candidates = [
            node for node in self.G_nx.nodes()
            if node != researcher_id 
            and node not in src_neighbors
            and not should_exclude(node)
        ]
    
        print(f"🔍 [GET_RECOMMENDATIONS] {len(candidates)} candidats après filtrage hubs")
    
        if len(candidates) == 0:
            print("⚠️ [GET_RECOMMENDATIONS] Aucun candidat, fallback")
            return self._fallback_for_isolated(top_k)
        
        # ========== 2. SCORES ML BRUTS ==========
        candidate_indices = [self.node_to_idx[c] for c in candidates]
        
        with torch.no_grad():
            edge_index = torch.tensor(
                [[src_idx] * len(candidate_indices), candidate_indices],
                dtype=torch.long
            ).to(self.device)
            ml_scores = self.predictor(self.embeddings, edge_index).cpu().numpy()
        
        print(f"✅ [GET_RECOMMENDATIONS] Scores ML calculés: min={ml_scores.min():.3f}, max={ml_scores.max():.3f}")
        
        # ========== 3. CALCUL SCORE FINAL ==========
        max_degree = max(all_degrees)
        
        results = []
        for idx, candidate in enumerate(candidates):
            deg = self.G_nx.degree(candidate)
            
            # Vérifier si dans cache
            cache_entry = self.metrics_cache.get(candidate)
            if cache_entry:
                pr = cache_entry["pagerank"]
                comm = self.partition.get(candidate, -1)
            else:
                # Fallback
                pr = 1.0 / len(self.G_nx.nodes())
                comm = -1
            
            # Voisins communs
            common_neighbors = len(src_neighbors & set(self.G_nx.neighbors(candidate)))
            
            ml_score = float(ml_scores[idx])
                
            # ========== PÉNALITÉS ANTI-HUB ==========
            degree_ratio = deg / max_degree
            degree_pct = sum(1 for d in all_degrees if d < deg) / len(all_degrees)
            
            # Pénalité exponentielle pour les top percentiles
            if degree_pct > 0.95:  # Top 5%
                hub_penalty = 0.4 + (degree_ratio * 0.3)  # 0.4 à 0.7
            elif degree_pct > 0.90:  # Top 10%
                hub_penalty = 0.25 + (degree_ratio * 0.2)  # 0.25 à 0.45
            elif degree_pct > 0.75:  # Top 25%
                hub_penalty = 0.15 * degree_ratio  # 0.11 à 0.15
            elif degree_pct > 0.50:  # Top 50%
                hub_penalty = 0.08 * degree_ratio  # 0.04 à 0.08
            else:
                hub_penalty = 0.03 * degree_ratio  # Minimal
            
            # ========== BONUS DIVERSITÉ ==========
            comm_bonus = 0.05 if comm != src_community else 0
            cn_bonus = min(common_neighbors * 0.01, 0.03)
            degree_similarity = 1 - abs(deg - src_degree) / max_degree
            balance_bonus = 0.03 * degree_similarity if degree_similarity > 0.5 else 0
            
            # ========== SCORE FINAL ==========
            final_score = (
                ml_score 
                - hub_penalty 
                + comm_bonus 
                + cn_bonus 
                + balance_bonus
            )
            
            results.append({
                "researcher_id": candidate,
                "ml_score": round(ml_score, 6),
                "collaboration_score": round(final_score, 6),
                "degree": int(deg),
                "degree_percentile": round(
                    sum(1 for d in all_degrees if d < deg) / len(all_degrees) * 100, 
                1
                ),
                "same_community": bool(comm == src_community),
                "community_id": int(comm),
                "pagerank": float(pr),
                "common_neighbors": int(common_neighbors),
                "penalties_applied": {
                    "hub_penalty": round(float(hub_penalty), 4),
                    "comm_bonus": round(float(comm_bonus), 4),
                    "cn_bonus": round(float(cn_bonus), 4),
                    "balance_bonus": round(float(balance_bonus), 4)
                }
            })
    
        # ========== 4. TRI DESCENDANT ==========
        results.sort(key=lambda x: x["collaboration_score"], reverse=True)
        
        print(f"📊 [GET_RECOMMENDATIONS] Top 3 scores: {[r['collaboration_score'] for r in results[:3]]}")
    
        # ========== 5. DIVERSIFICATION FORCÉE ==========
        final = []
        communities_seen = {}
        degree_buckets_seen = {}
    
        # Limites par communauté et bucket de degré
        max_per_community = max(2, top_k // 5)
        max_per_degree_bucket = max(3, top_k // 3)
    
        for r in results:
            comm = r["community_id"]
        
            # Bucket de degré (bas/moyen/haut)
            deg_pct = r["degree_percentile"]
            if deg_pct > 75:
                deg_bucket = "high"
            elif deg_pct > 50:
                deg_bucket = "medium"
            else:
                deg_bucket = "low"
        
            # Compter occurrences
            communities_seen.setdefault(comm, 0)
            degree_buckets_seen.setdefault(deg_bucket, 0)
            
            # Vérifier limites de diversité
            if communities_seen[comm] >= max_per_community:
                continue
                
            if degree_buckets_seen[deg_bucket] >= max_per_degree_bucket:
                continue
            
            # Ajouter à la liste finale
            final.append(r)
            communities_seen[comm] += 1
            degree_buckets_seen[deg_bucket] += 1
            
            if len(final) == top_k:
                break
        
        # ========== 6. COMPLÉTER SI INSUFFISANT ==========
        if len(final) < top_k:
            remaining = [r for r in results if r not in final]
            import random
            random.shuffle(remaining)
            
            for r in remaining:
                if len(final) >= top_k:
                    break
                final.append(r)
        
        # ========== 7. AJOUTER RANKING ==========
        for i, f in enumerate(final):
            f["rank"] = i + 1
        
        print(f"✅ [GET_RECOMMENDATIONS] {len(final)} recommandations finales")
        
        return final[:top_k]
        
    def get_author_analytics(self, author_id: str) -> Dict:
        """Récupérer métriques détaillées d'un auteur"""
        # Convertir en int si nécessaire
        try:
            author_id = int(author_id)
        except ValueError:
            pass
        
        if author_id not in self.G_full:
            raise ValueError(f"Auteur '{author_id}' inconnu dans la base de données")
        
        # Si l'auteur est dans le modèle ML
        if author_id in self.node_to_idx and author_id in self.G_nx:
            metrics = self.metrics_cache[author_id]
            collaborators = list(self.G_nx.neighbors(author_id))
            community_id = metrics['community']
            community_members = [
                node for node, comm in self.partition.items() 
                if comm == community_id
            ]
            
            ego = nx.ego_graph(self.G_nx, author_id, radius=1)
            
            collab_by_community = {}
            for collab in collaborators:
                comm = self.partition.get(collab, -1)
                collab_by_community[comm] = collab_by_community.get(comm, 0) + 1
            
            all_pageranks = [m['pagerank'] for m in self.metrics_cache.values()]
            pagerank_percentile = (
                sum(1 for pr in all_pageranks if pr < metrics['pagerank']) / 
                len(all_pageranks) * 100
            )
            
            return {
                'author_id': author_id,
                'in_ml_model': True,
                'influence_metrics': {
                    'degree': metrics['degree'],
                    'degree_centrality': round(metrics['degree_centrality'], 4),
                    'betweenness_centrality': round(metrics['betweenness_centrality'], 4),
                    'closeness_centrality': round(metrics['closeness_centrality'], 4),
                    'pagerank': round(metrics['pagerank'], 6),
                    'pagerank_percentile': round(pagerank_percentile, 2),
                    'clustering_coefficient': round(metrics['clustering_coefficient'], 4)
                },
                'network_stats': {
                    'total_collaborators': len(collaborators),
                    'ego_network_size': ego.number_of_nodes(),
                    'ego_network_density': round(nx.density(ego), 4),
                    'community_id': community_id,
                    'community_size': len(community_members),
                    'collaborators_by_community': collab_by_community
                },
                'top_collaborators': [
                    {
                        'author_id': collab,
                        'degree': self.G_nx.degree(collab),
                        'pagerank': round(self.metrics_cache[collab]['pagerank'], 6),
                        'same_community': self.partition.get(collab, -1) == community_id
                    }
                    for collab in sorted(
                        collaborators, 
                        key=lambda x: self.metrics_cache[x]['pagerank'], 
                        reverse=True
                    )[:5]
                ],
                'collaboration_diversity': {
                    'unique_communities': len(collab_by_community),
                    'primary_community_ratio': round(
                        max(collab_by_community.values()) / len(collaborators) if collaborators else 0,
                        4
                    )
                }
            }
        
        # Si l'auteur est hors modèle ML
        else:
            collaborators = list(self.G_full.neighbors(author_id))
            degree = self.G_full.degree(author_id)
            
            return {
                'author_id': author_id,
                'in_ml_model': False,
                'influence_metrics': {
                    'degree': degree,
                    'degree_centrality': None,
                    'betweenness_centrality': None,
                    'closeness_centrality': None,
                    'pagerank': None,
                    'pagerank_percentile': None,
                    'clustering_coefficient': None
                },
                'network_stats': {
                    'total_collaborators': len(collaborators),
                    'ego_network_size': None,
                    'ego_network_density': None,
                    'community_id': None,
                    'community_size': None,
                    'collaborators_by_community': {}
                },
                'top_collaborators': [
                    {
                        'author_id': collab,
                        'degree': self.G_full.degree(collab),
                        'pagerank': None,
                        'same_community': False
                    }
                    for collab in sorted(
                        collaborators, 
                        key=lambda x: self.G_full.degree(x), 
                        reverse=True
                    )[:5]
                ],
                'collaboration_diversity': {
                    'unique_communities': None,
                    'primary_community_ratio': None
                }
            }

def _compute_missing_metrics(self, missing_nodes):
    """Calculer les métriques manquantes"""
    print(f"🔧 Calcul des métriques pour {len(missing_nodes)} nœuds manquants...")
    
    for node in missing_nodes:
        if node in self.G_nx:
            # Calculer les métriques pour ce nœud spécifique
            degree = self.G_nx.degree(node)
            clustering = nx.clustering(self.G_nx, node)
            
            # Pour les autres métriques, vous pouvez utiliser des approximations
            self.metrics_cache[node] = {
                'degree': degree,
                'degree_centrality': degree / (self.G_nx.number_of_nodes() - 1),
                'betweenness_centrality': 0.0,  # Approx
                'closeness_centrality': 0.0,  # Approx
                'pagerank': 1.0 / self.G_nx.number_of_nodes(),  # Valeur initiale
                'clustering_coefficient': clustering,
                'community': self.partition.get(node, -1)
            }

    def _fallback_for_isolated(self, top_k):  # CORRECTION : Même niveau
        """Fallback pour auteurs isolés : proposer top auteurs influents du graphe."""
        
        # Top par degree
        sorted_degree = sorted(
            self.G_full.degree(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
    
        # Top par PageRank
        sorted_pr = sorted(
            self.metrics_cache.items(),
            key=lambda x: x[1]['pagerank'],
            reverse=True
        )[:top_k]
    
        # Combiner résultats
        combined = {}
    
        for node, deg in sorted_degree:
            combined[node] = {
                "degree": deg,
                "pagerank": self.metrics_cache.get(node, {}).get("pagerank", 0),
                "reason": "Top-degree global (fallback)"
            }
    
        for node, stats in sorted_pr:
            if node not in combined:
                combined[node] = {
                    "degree": self.G_full.degree(node),
                    "pagerank": stats["pagerank"],
                    "reason": "Top PageRank global (fallback)"
                }
    
        # Retourner top_k uniques
        top_nodes = list(combined.keys())[:top_k]
    
        return [
            {
                "researcher_id": node,
                "degree": combined[node]["degree"],
                "pagerank": combined[node]["pagerank"],
                "reason": combined[node]["reason"]
            }
            for node in top_nodes
        ]

    def _normalize_id(self, author_id):  # CORRECTION : Même niveau
        """Convertit un ID string → int si possible."""
        try:
            return int(author_id)
        except:
            return author_id  # fallback (rare)

    def get_recommendations(self, researcher_id: str, top_k: int = 10, 
                            strategy: str = "ml_only", filter_hubs: bool = True,
                            hub_threshold_percentile: int = 95):
        """
        Version corrigée avec vraie diversification
        
        Args:
            hub_threshold_percentile: Percentile au-dessus duquel exclure les hubs
                                      95 = exclure top 5% (par défaut)
                                      90 = exclure top 10% (plus strict)
                                      98 = exclure top 2% (moins strict)
        """
        
        # CONVERTIR L'ID EN INT SI POSSIBLE
        try:
            # Essayer de convertir en int
            researcher_id_int = int(researcher_id)
            
            # Chercher d'abord comme int, puis comme string
            if researcher_id_int in self.node_to_idx:
                researcher_id = researcher_id_int
                print(f"🔍 ID converti en int: {researcher_id}")
            elif researcher_id in self.node_to_idx:
                print(f"🔍 ID utilisé comme string: {researcher_id}")
            else:
                # Essayer les deux formats
                if str(researcher_id) in self.node_to_idx:
                    researcher_id = str(researcher_id)
                    print(f"🔍 ID converti en string: {researcher_id}")
                else:
                    raise ValueError(f"Auteur {researcher_id} inconnu dans le modèle")
        except ValueError:
            # Si pas convertible, garder tel quel
            if researcher_id not in self.node_to_idx:
                # Essayer comme string
                if str(researcher_id) in self.node_to_idx:
                    researcher_id = str(researcher_id)
                else:
                    raise ValueError(f"Auteur {researcher_id} inconnu dans le modèle")
        
        print(f"🎯 Recherche recommandations pour ID: {researcher_id} (type: {type(researcher_id)})")
        
        if researcher_id not in self.node_to_idx:
            raise ValueError(f"Auteur {researcher_id} inconnu dans le modèle")
    
        if researcher_id not in self.G_nx or self.G_nx.degree(researcher_id) == 0:
            return self._fallback_for_isolated(top_k)
    
        src_idx = self.node_to_idx[researcher_id]
        src_neighbors = set(self.G_nx.neighbors(researcher_id))
        src_community = self.partition.get(researcher_id, -1)
        src_degree = self.G_nx.degree(researcher_id)
    
        # ========== 1. FILTRAGE STRICT DES HUBS ==========
        all_degrees = [self.G_nx.degree(n) for n in self.G_nx.nodes()]
        
        # Seuils percentiles
        p_threshold = np.percentile(all_degrees, hub_threshold_percentile)
        p90 = np.percentile(all_degrees, 90)
        median_deg = np.median(all_degrees)
        
        def should_exclude(node):
            deg = self.G_nx.degree(node)
            
            # Exclure selon le seuil paramétrable
            if deg > p_threshold:
                return True
            
            # Exclure top 10% si l'auteur source est dans la moitié inférieure
            if src_degree < median_deg and deg > p90:
                return True
                
            return False
    
        candidates = [
            node for node in self.G_nx.nodes()
            if node != researcher_id 
            and node not in src_neighbors
            and not should_exclude(node)
        ]
    
        if len(candidates) == 0:
            return self._fallback_for_isolated(top_k)
    
        # ========== 2. SCORES ML BRUTS ==========
        candidate_indices = [self.node_to_idx[c] for c in candidates]
    
        with torch.no_grad():
            edge_index = torch.tensor(
                [[src_idx] * len(candidate_indices), candidate_indices],
                dtype=torch.long
            ).to(self.device)
            ml_scores = self.predictor(self.embeddings, edge_index).cpu().numpy()
    
        # ========== 3. CALCUL SCORE FINAL AVEC PÉNALITÉS ==========
        max_degree = max(all_degrees)
        
        results = []
        for idx, candidate in enumerate(candidates):
            deg = self.G_nx.degree(candidate)
            cache_entry = self.metrics_cache.get(candidate)
            if cache_entry:
                pr = cache_entry["pagerank"]
                comm = self.partition.get(candidate, -1)
            else:
                pr = 0.0
                comm = -1
                clustering = nx.clustering(self.G_nx, candidate) if candidate in self.G_nx else 0.0

            
            # Voisins communs
            common_neighbors = len(src_neighbors & set(self.G_nx.neighbors(candidate)))
            
            ml_score = float(ml_scores[idx])
            
            # ========== PÉNALITÉS ANTI-HUB ULTRA-AGRESSIVES ==========
            # Pour contrer des ML scores de 0.95+, il faut des pénalités MASSIVES
            degree_ratio = deg / max_degree
            degree_pct = sum(1 for d in all_degrees if d < deg) / len(all_degrees)
            
            # Pénalité exponentielle pour les top percentiles
            if degree_pct > 0.95:  # Top 5%
                hub_penalty = 0.4 + (degree_ratio * 0.3)  # 0.4 à 0.7
            elif degree_pct > 0.90:  # Top 10%
                hub_penalty = 0.25 + (degree_ratio * 0.2)  # 0.25 à 0.45
            elif degree_pct > 0.75:  # Top 25%
                hub_penalty = 0.15 * degree_ratio  # 0.11 à 0.15
            elif degree_pct > 0.50:  # Top 50%
                hub_penalty = 0.08 * degree_ratio  # 0.04 à 0.08
            else:
                hub_penalty = 0.03 * degree_ratio  # Minimal
            
            # ========== BONUS DIVERSITÉ ==========
            # Bonus si communauté différente (encourage exploration)
            comm_bonus = 0.05 if comm != src_community else 0
            
            # Bonus pour voisins communs (mais limité)
            cn_bonus = min(common_neighbors * 0.01, 0.03)
            
            # Bonus si degré similaire (collaboration plus équilibrée)
            degree_similarity = 1 - abs(deg - src_degree) / max_degree
            balance_bonus = 0.03 * degree_similarity if degree_similarity > 0.5 else 0
            
            # ========== SCORE FINAL ==========
            final_score = (
                ml_score 
                - hub_penalty 
                + comm_bonus 
                + cn_bonus 
                + balance_bonus
            )
            
            results.append({
                "researcher_id": candidate,
                "ml_score": round(ml_score, 6),
                "collaboration_score": round(final_score, 6),
                "degree": int(deg),
                "degree_percentile": round(
                    sum(1 for d in all_degrees if d < deg) / len(all_degrees) * 100, 
                    1
                ),
                "same_community": bool(comm == src_community),
                "community_id": int(comm),
                "pagerank": float(pr),
                "common_neighbors": int(common_neighbors),
                "penalties_applied": {
                    "hub_penalty": round(float(hub_penalty), 4),
                    "comm_bonus": round(float(comm_bonus), 4),
                    "cn_bonus": round(float(cn_bonus), 4),
                    "balance_bonus": round(float(balance_bonus), 4)
                }
            })
    
        # ========== 4. TRI DESCENDANT ==========
        results.sort(key=lambda x: x["collaboration_score"], reverse=True)
    
        # ========== 5. DIVERSIFICATION FORCÉE ==========
        final = []
        communities_seen = {}
        degree_buckets_seen = {}
        
        # Limites par communauté et bucket de degré
        max_per_community = max(2, top_k // 5)  # Max 20% d'une même communauté
        max_per_degree_bucket = max(3, top_k // 3)  # Max 33% d'un même niveau de degré
        
        for r in results:
            comm = r["community_id"]
            
            # Bucket de degré (bas/moyen/haut)
            deg_pct = r["degree_percentile"]
            if deg_pct > 75:
                deg_bucket = "high"
            elif deg_pct > 50:
                deg_bucket = "medium"
            else:
                deg_bucket = "low"
            
            # Compter occurrences
            communities_seen.setdefault(comm, 0)
            degree_buckets_seen.setdefault(deg_bucket, 0)
            
            # Vérifier limites de diversité
            if communities_seen[comm] >= max_per_community:
                continue
                
            if degree_buckets_seen[deg_bucket] >= max_per_degree_bucket:
                continue
            
            # Ajouter à la liste finale
            final.append(r)
            communities_seen[comm] += 1
            degree_buckets_seen[deg_bucket] += 1
            
            if len(final) == top_k:
                break
    
        # ========== 6. COMPLÉTER SI INSUFFISANT ==========
        if len(final) < top_k:
            remaining = [r for r in results if r not in final]
            
            # Prioriser diversité sur le reste
            import random
            random.shuffle(remaining)
            
            for r in remaining:
                if len(final) >= top_k:
                    break
                final.append(r)
    
        # ========== 7. AJOUTER RANKING ==========
        for i, f in enumerate(final):
            f["rank"] = i + 1
    
        return final[:top_k]

    def get_author_analytics(self, author_id: str) -> Dict:
        """Récupérer métriques détaillées d'un auteur"""
        
        # Convertir en int si nécessaire
        try:
            author_id = int(author_id)
        except ValueError:
            pass
        
        if author_id not in self.G_full:
            raise ValueError(f"Auteur '{author_id}' inconnu dans la base de données")
        
        # Si l'auteur est dans le modèle ML
        if author_id in self.node_to_idx and author_id in self.G_nx:
            metrics = self.metrics_cache[author_id]
            collaborators = list(self.G_nx.neighbors(author_id))
            community_id = metrics['community']
            community_members = [
                node for node, comm in self.partition.items() 
                if comm == community_id
            ]
            
            ego = nx.ego_graph(self.G_nx, author_id, radius=1)
            
            collab_by_community = {}
            for collab in collaborators:
                comm = self.partition.get(collab, -1)
                collab_by_community[comm] = collab_by_community.get(comm, 0) + 1
            
            all_pageranks = [m['pagerank'] for m in self.metrics_cache.values()]
            pagerank_percentile = (
                sum(1 for pr in all_pageranks if pr < metrics['pagerank']) / 
                len(all_pageranks) * 100
            )
            
            return {
                'author_id': author_id,
                'in_ml_model': True,
                'influence_metrics': {
                    'degree': metrics['degree'],
                    'degree_centrality': round(metrics['degree_centrality'], 4),
                    'betweenness_centrality': round(metrics['betweenness_centrality'], 4),
                    'closeness_centrality': round(metrics['closeness_centrality'], 4),
                    'pagerank': round(metrics['pagerank'], 6),
                    'pagerank_percentile': round(pagerank_percentile, 2),
                    'clustering_coefficient': round(metrics['clustering_coefficient'], 4)
                },
                'network_stats': {
                    'total_collaborators': len(collaborators),
                    'ego_network_size': ego.number_of_nodes(),
                    'ego_network_density': round(nx.density(ego), 4),
                    'community_id': community_id,
                    'community_size': len(community_members),
                    'collaborators_by_community': collab_by_community
                },
                'top_collaborators': [
                    {
                        'author_id': collab,
                        'degree': self.G_nx.degree(collab),
                        'pagerank': round(self.metrics_cache[collab]['pagerank'], 6),
                        'same_community': self.partition.get(collab, -1) == community_id
                    }
                    for collab in sorted(
                        collaborators, 
                        key=lambda x: self.metrics_cache[x]['pagerank'], 
                        reverse=True
                    )[:5]
                ],
                'collaboration_diversity': {
                    'unique_communities': len(collab_by_community),
                    'primary_community_ratio': round(
                        max(collab_by_community.values()) / len(collaborators) if collaborators else 0,
                        4
                    )
                }
            }
        
        # ✅ Si l'auteur est hors modèle ML → métriques basiques du graphe complet
        else:
            collaborators = list(self.G_full.neighbors(author_id))
            degree = self.G_full.degree(author_id)
            
            return {
                'author_id': author_id,
                'in_ml_model': False,
                'influence_metrics': {
                    'degree': degree,
                    'degree_centrality': None,
                    'betweenness_centrality': None,
                    'closeness_centrality': None,
                    'pagerank': None,
                    'pagerank_percentile': None,
                    'clustering_coefficient': None
                },
                'network_stats': {
                    'total_collaborators': len(collaborators),
                    'ego_network_size': None,
                    'ego_network_density': None,
                    'community_id': None,
                    'community_size': None,
                    'collaborators_by_community': {}
                },
                'top_collaborators': [
                    {
                        'author_id': collab,
                        'degree': self.G_full.degree(collab),
                        'pagerank': None,
                        'same_community': False
                    }
                    for collab in sorted(
                        collaborators, 
                        key=lambda x: self.G_full.degree(x), 
                        reverse=True
                    )[:5]
                ],
                'collaboration_diversity': {
                    'unique_communities': None,
                    'primary_community_ratio': None
                }
            }
       
def export_api_data():
    """Exporter les données au format JSON pour le frontend"""
    import json
    from pathlib import Path
    
    output_dir = Path("../results")
    output_dir.mkdir(exist_ok=True)
    
    print("📤 Exportation des données pour le frontend...")
    
    # 1. Communautés
    if analytics.partition:
        communities_dict = {}
        for node, comm_id in analytics.partition.items():
            if comm_id not in communities_dict:
                communities_dict[comm_id] = []
            communities_dict[comm_id].append(node)
        
        communities_list = []
        for comm_id, nodes in communities_dict.items():
            subgraph = analytics.G_nx.subgraph(nodes)
            top_members = []
            
            for node in nodes:
                if node in analytics.metrics_cache:
                    metrics = analytics.metrics_cache[node]
                    top_members.append({
                        "author_id": str(node),
                        "name": f"Author {node}",
                        "publication_count": metrics.get("degree", 0),
                        "centrality_score": metrics.get("degree_centrality", 0)
                    })
            
            top_members.sort(key=lambda x: x["centrality_score"], reverse=True)
            
            communities_list.append({
                "id": comm_id,
                "name": f"Research Community {comm_id}",
                "size": len(nodes),
                "density": round(nx.density(subgraph) if subgraph.number_of_edges() > 0 else 0, 3),
                "topics": ["Computer Science", "Physics", "Mathematics"][:comm_id % 3 + 1],
                "top_members": top_members[:10],
                "description": f"Community of {len(nodes)} researchers"
            })
        
        communities_list.sort(key=lambda x: x["size"], reverse=True)
        
        with open(output_dir / "communities_api.json", 'w', encoding='utf-8') as f:
            json.dump({"communities": communities_list}, f, indent=2)
        print(f"✅ {len(communities_list)} communautés exportées")
    
    # 2. Author map
    if analytics.partition:
        author_map = {}
        for node, comm_id in analytics.partition.items():
            author_map[str(node)] = {
                "name": f"Author {node}",
                "community_id": comm_id
            }
        
        with open(output_dir / "author_community_map.json", 'w', encoding='utf-8') as f:
            json.dump(author_map, f, indent=2)
        print(f"✅ {len(author_map)} auteurs mappés")
    
    # 3. Summary
    if analytics.partition and communities_list:
        summary = {
            "total_communities": len(communities_list),
            "total_authors": sum(c["size"] for c in communities_list),
            "average_size": sum(c["size"] for c in communities_list) / len(communities_list),
            "average_density": sum(c["density"] for c in communities_list) / len(communities_list),
            "largest_community": {
                "id": communities_list[0]["id"] if communities_list else None,
                "name": communities_list[0]["name"] if communities_list else None,
                "size": communities_list[0]["size"] if communities_list else 0
            }
        }
        
        with open(output_dir / "summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        print("✅ Résumé exporté")

# ============= INITIALISATION FASTAPI =============
analytics = ResearcherAnalytics()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charger le modèle au démarrage et exporter les données"""
    import os
    import sys
    from pathlib import Path
    
    print("=" * 60)
    print("🚀 Démarrage de l'API ML pour Research Collaboration")
    print("=" * 60)
    
    # ========== 1. VÉRIFICATION DES CHEMINS ==========
    BASE_DIR = Path(__file__).parent.parent
    
    print(f"\n📁 Répertoire de base: {BASE_DIR}")
    
    # CHANGEMENT ICI : Charger d'abord le modèle étendu
    model_path = BASE_DIR / "models" / "link_prediction_model_extended.pkl"
    
    if model_path.exists():
        print(f"✅ Modèle étendu trouvé: {model_path}")
    else:
        # Fallback au modèle original
        model_path = BASE_DIR / "models" / "link_prediction_model.pkl"
        if model_path.exists():
            print(f"⚠️  Modèle étendu non trouvé, utilisation du modèle original: {model_path}")
        else:
            print("❌ Aucun modèle trouvé!")
            model_path = BASE_DIR / "models" / "test_model.pt"
            create_test_model(model_path)
    
    # Chemin de configuration
    config_path = BASE_DIR / "config.yaml"
    if not config_path.exists():
        create_default_config(config_path)
    
    # ========== 2. CHARGEMENT DU MODÈLE ==========
    print("\n📥 Chargement du modèle étendu...")
    try:
        analytics.load_model(model_path=str(model_path), config_path=str(config_path))
        print("✅ Modèle étendu chargé avec succès!")
        
        # Afficher les statistiques
        print(f"\n📊 STATISTIQUES DU MODÈLE:")
        print(f"   • Auteurs dans le modèle: {len(analytics.node_to_idx)}")
        print(f"   • Embeddings: {analytics.embeddings.shape}")
        print(f"   • Communautés détectées: {len(set(analytics.partition.values()))}")
        print(f"   • Arêtes dans le graphe: {analytics.G_nx.number_of_edges()}")
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")
        print("Traceback complet:")
        import traceback
        traceback.print_exc()
        print("\n⚠️  Mode dégradé: initialisation avec données minimales")
        initialize_minimal_analytics()
    
    # ========== 3. CONNEXION NEO4J ==========
    print("\n🔗 Connexion à Neo4j...")
    try:
        neo4j_client.connect()
        print("✅ Connexion Neo4j établie")
    except Exception as e:
        print(f"⚠️  Impossible de se connecter à Neo4j: {e}")
        print("ℹ️  Les noms seront récupérés depuis le cache")
    
    # ========== 4. EXPORTATION DES DONNÉES ==========
    print("\n📤 Exportation des données pour le frontend...")
    try:
        export_data_for_frontend()
        print("✅ Données exportées avec succès!")
    except Exception as e:
        print(f"⚠️  Erreur lors de l'export des données: {e}")
        create_mock_data_files()
    
    print("\n" + "=" * 60)
    print("✅ API ML prête à recevoir des requêtes")
    print("📡 Endpoints disponibles sur http://localhost:8000")
    print("=" * 60 + "\n")
    
    yield
    
    # Nettoyage
    neo4j_client.close()
    print("\n🛑 Arrêt de l'API...")

def create_test_model(model_path: Path):
    """Créer un modèle minimal pour les tests"""
    import torch
    import yaml
    
    print(f"🔧 Création d'un modèle de test: {model_path}")
    
    # Créer une config par défaut
    config = {
        'model': {
            'embedding_dim': 64,
            'hidden_dim': 128,
            'dropout': 0.5
        },
        'training': {
            'epochs': 10,
            'lr': 0.001
        }
    }
    
    # Créer un modèle minimal
    test_model = {
        'encoder_state_dict': {},
        'predictor_state_dict': {},
        'embeddings': torch.randn(100, 64),
        'node_to_idx': {f'test_{i}': i for i in range(100)},
        'config': config
    }
    
    model_path.parent.mkdir(exist_ok=True, parents=True)
    torch.save(test_model, model_path)

def create_default_config(config_path: Path):
    """Créer une configuration par défaut"""
    import yaml
    
    config = {
        'neo4j': {
            'uri': 'bolt://localhost:7687',
            'user': 'neo4j',
            'password': 'password',
            'database': 'neo4j'
        },
        'model': {
            'embedding_dim': 64,
            'hidden_channels': 128,
            'dropout': 0.5
        }
    }
    
    config_path.parent.mkdir(exist_ok=True, parents=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def initialize_minimal_analytics():
    """Initialiser analytics avec des données minimales"""
    import networkx as nx
    import torch
    
    print("🔧 Initialisation des données minimales...")
    
    # Créer un graphe minimal
    G = nx.karate_club_graph()
    
    # Renommer les nœuds pour correspondre à des IDs d'auteurs
    mapping = {i: f"test_{i}" for i in range(G.number_of_nodes())}
    G = nx.relabel_nodes(G, mapping)
    
    analytics.G_nx = G
    analytics.G_full = G
    
    # Créer node_to_idx
    analytics.node_to_idx = {node: i for i, node in enumerate(G.nodes())}
    analytics.idx_to_node = {v: k for k, v in analytics.node_to_idx.items()}
    
    # Créer embeddings aléatoires
    analytics.embeddings = torch.randn(len(analytics.node_to_idx), 64)
    
    # Créer partition de communautés
    analytics.partition = {node: i % 3 for i, node in enumerate(G.nodes())}
    
    # Initialiser metrics_cache
    analytics.metrics_cache = {}
    for node in G.nodes():
        analytics.metrics_cache[node] = {
            'degree': G.degree(node),
            'degree_centrality': 0.1,
            'pagerank': 0.01,
            'clustering_coefficient': 0.5,
            'community': analytics.partition.get(node, -1)
        }
    
    print(f"✅ Données minimales créées: {len(analytics.node_to_idx)} nœuds")

def export_data_for_frontend():
    """Exporter les données pour le frontend avec vrais noms"""
    import json
    from pathlib import Path
    
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    print("📤 Exportation des données avec noms réels...")
    
    # 1. Récupérer tous les IDs d'auteurs uniques
    all_author_ids = set()
    
    if hasattr(analytics, 'partition') and analytics.partition:
        for node in analytics.partition.keys():
            all_author_ids.add(str(node))
    
    print(f"📊 Récupération des noms pour {len(all_author_ids)} auteurs...")
    
    # 2. Récupérer les noms depuis Neo4j
    author_names = {}
    try:
        # Récupérer par lots de 1000 pour éviter les timeouts
        ids_list = list(all_author_ids)
        for i in range(0, len(ids_list), 1000):
            batch = ids_list[i:i+1000]
            batch_names = neo4j_client.get_author_names(batch)
            author_names.update(batch_names)
            print(f"  Lot {i//1000 + 1}: {len(batch_names)} noms récupérés")
    except Exception as e:
        print(f"⚠️  Erreur récupération noms: {e}")
        # Fallback: utiliser des noms génériques
        author_names = {aid: f"Author {aid}" for aid in all_author_ids}
    
    print(f"✅ {len(author_names)} noms récupérés")
    
    # 3. Exporter les communautés avec vrais noms
    if hasattr(analytics, 'partition') and analytics.partition:
        print("📊 Exportation des communautés...")
        
        communities_dict = {}
        for node, comm_id in analytics.partition.items():
            node_str = str(node)
            if comm_id not in communities_dict:
                communities_dict[comm_id] = []
            communities_dict[comm_id].append(node_str)
        
        # Formater les données
        communities_list = []
        for comm_id, nodes in communities_dict.items():
            subgraph = analytics.G_nx.subgraph(nodes)
            
            # Trouver les top membres par centralité
            top_members_info = []
            for node in nodes:
                if node in analytics.metrics_cache:
                    metrics = analytics.metrics_cache[node]
                    top_members_info.append({
                        "author_id": node,
                        "centrality": metrics.get("degree_centrality", 0),
                        "degree": metrics.get("degree", 0)
                    })
            
            # Trier par centralité
            top_members_info.sort(key=lambda x: x["centrality"], reverse=True)
            
            # Formater avec vrais noms
            top_members = []
            for member in top_members_info[:10]:  # Top 10
                author_id = member["author_id"]
                top_members.append({
                    "author_id": author_id,
                    "name": author_names.get(author_id, f"Author {author_id}"),
                    "publication_count": member["degree"],
                    "centrality_score": member["centrality"]
                })
            
            # Nommer les communautés intelligemment
            community_name = generate_community_name(comm_id, top_members)
            
            communities_list.append({
                "id": comm_id,
                "name": community_name,
                "size": len(nodes),
                "density": round(nx.density(subgraph) if subgraph.number_of_edges() > 0 else 0, 3),
                "topics": detect_topics_from_members(top_members),
                "top_members": top_members,
                "description": f"Communauté de {len(nodes)} chercheurs en {', '.join(detect_topics_from_members(top_members)[:2])}"
            })
        
        # Trier par taille
        communities_list.sort(key=lambda x: x["size"], reverse=True)
        
        # Sauvegarder
        with open(results_dir / "communities_api.json", 'w', encoding='utf-8') as f:
            json.dump({"communities": communities_list}, f, indent=2)
        
        print(f"✅ {len(communities_list)} communautés exportées avec vrais noms")
    
    # 4. Exporter author map avec vrais noms
    if hasattr(analytics, 'partition') and analytics.partition:
        print("🗺️  Exportation de la map auteur->communauté...")
        
        author_map = {}
        for node, comm_id in analytics.partition.items():
            node_str = str(node)
            author_map[node_str] = {
                "name": author_names.get(node_str, f"Author {node_str}"),
                "community_id": comm_id
            }
        
        with open(results_dir / "author_community_map.json", 'w', encoding='utf-8') as f:
            json.dump(author_map, f, indent=2)
        
        print(f"✅ {len(author_map)} auteurs mappés avec vrais noms")
    
    # 5. Exporter summary
    if communities_list:
        print("📈 Exportation du résumé...")
        
        summary = {
            "total_communities": len(communities_list),
            "total_authors": sum(c["size"] for c in communities_list),
            "average_size": sum(c["size"] for c in communities_list) / len(communities_list),
            "average_density": sum(c["density"] for c in communities_list) / len(communities_list),
            "largest_community": {
                "id": communities_list[0]["id"],
                "name": communities_list[0]["name"],
                "size": communities_list[0]["size"]
            },
            "description": "Communautés détectées à partir du réseau de collaborations scientifiques"
        }
        
        with open(results_dir / "summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print("✅ Résumé exporté")

def generate_community_name(community_id, top_members):
    """Génère un nom intelligent pour la communauté"""
    if not top_members:
        return f"Research Community {community_id}"
    
    # Essayer de déduire le domaine des noms
    names = [member["name"] for member in top_members[:3]]
    
    # Keywords pour différents domaines
    domain_keywords = {
        "AI": ["hinton", "bengio", "lecun", "goodfellow", "silver"],
        "Deep Learning": ["hochreiter", "schmidhuber", "krizhevsky", "sutskever"],
        "Physics": ["einstein", "feynman", "hawking", "schrödinger", "heisenberg"],
        "Mathematics": ["gauss", "euler", "newton", "leibniz", "ramanujan"],
        "Computer Science": ["dijkstra", "knuth", "turing", "von neumann"],
        "Neuroscience": ["cajal", "hubel", "wiesel", "kandel"],
        "Biology": ["darwin", "pasteur", "mendel", "watson", "crick"]
    }
    
    for domain, keywords in domain_keywords.items():
        for name in names:
            if any(keyword.lower() in name.lower() for keyword in keywords):
                return f"{domain} Research Community {community_id}"
    
    # Sinon, utiliser le top auteur
    top_name = top_members[0]["name"].split()[-1]  # Dernier nom
    return f"{top_name} Research Network"

def detect_topics_from_members(top_members):
    """Détecte les domaines à partir des noms des membres"""
    # Logique simplifiée - à améliorer avec vraies données
    topics = ["Computer Science", "Mathematics", "Physics"]
    
    # Analyser les noms pour déduire les domaines
    for member in top_members[:5]:
        name_lower = member["name"].lower()
        if any(word in name_lower for word in ["ai", "learning", "neural", "network"]):
            if "AI" not in topics:
                topics.append("AI")
        elif any(word in name_lower for word in ["physic", "quantum", "particle"]):
            if "Physics" not in topics:
                topics.insert(0, "Physics")
    
    return topics[:3]  # Retourner max 3 topics

def create_mock_data_files():
    """Créer des fichiers de données mockés"""
    import json
    from pathlib import Path
    
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Données mockées
    mock_data = {
        "communities_api.json": {
            "communities": [
                {
                    "id": 1,
                    "name": "Deep Learning Research",
                    "size": 156,
                    "density": 0.82,
                    "topics": ["Computer Science", "AI", "Mathematics"],
                    "top_members": [
                        {
                            "author_id": "3308557",
                            "name": "Sepp Hochreiter",
                            "publication_count": 45,
                            "centrality_score": 0.95
                        },
                        {
                            "author_id": "34917892",
                            "name": "Djork-Arné Clevert",
                            "publication_count": 32,
                            "centrality_score": 0.88
                        }
                    ]
                }
            ]
        },
        "author_community_map.json": {
            "3308557": {"name": "Sepp Hochreiter", "community_id": 1},
            "34917892": {"name": "Djork-Arné Clevert", "community_id": 1}
        },
        "summary.json": {
            "total_communities": 5,
            "total_authors": 245,
            "average_size": 49,
            "average_density": 0.75,
            "largest_community": {"id": 1, "name": "Deep Learning Research", "size": 156}
        }
    }
    
    for filename, data in mock_data.items():
        with open(results_dir / filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    print("📦 Fichiers mockés créés dans results/")

app = FastAPI(
    title="Research Collaboration API",
    description="API ML pour recommandations de collaborations",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============= ENDPOINTS =============

@app.get("/")
def root():
    return {
        "api": "Research Collaboration ML API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": analytics.encoder is not None,
        "total_researchers": len(analytics.node_to_idx) if analytics.node_to_idx else 0,
        "device": str(analytics.device)
    }

@app.get("/recommendations/{author_id}")
def get_recommendations_endpoint(
    author_id: str = Path(..., description="ID de l'auteur"),
    top_k: int = Query(10, ge=1, le=50, description="Nombre de recommandations"),
    strategy: str = Query("ml_only", description="Stratégie: ml_only, hybrid, diverse"),
    filter_hubs: bool = Query(True, description="Exclure hubs (top 5% degree)")
):
    try:
        print(f"🎯 [RECOMMENDATIONS] Début pour author_id: {author_id}, type: {type(author_id)}")
        
        # Vérifier si l'auteur existe dans Neo4j d'abord
        try:
            print(f"🔍 [RECOMMENDATIONS] Vérification de l'auteur dans Neo4j...")
            test_names = neo4j_client.get_author_names([author_id])
            if author_id in test_names:
                print(f"✅ [RECOMMENDATIONS] Auteur trouvé dans Neo4j: {test_names[author_id]}")
            else:
                print(f"⚠️ [RECOMMENDATIONS] Auteur non trouvé dans Neo4j")
        except Exception as neo4j_error:
            print(f"⚠️ [RECOMMENDATIONS] Erreur vérification Neo4j: {neo4j_error}")
        
        # Obtenir les recommandations
        print(f"🔍 [RECOMMENDATIONS] Appel à analytics.get_recommendations...")
        recommendations = analytics.get_recommendations(
            author_id, top_k, strategy, filter_hubs
        )
        
        print(f"✅ [RECOMMENDATIONS] {len(recommendations)} recommandations obtenues")
        
        if not recommendations:
            print(f"⚠️ [RECOMMENDATIONS] Aucune recommandation obtenue")
            return {
                "author_id": author_id,
                "author_name": "Unknown",
                "strategy": strategy,
                "filter_hubs_applied": filter_hubs,
                "total_recommendations": 0,
                "recommendations": []
            }
        
        # Récupérer les IDs des auteurs recommandés
        author_ids = [str(rec['researcher_id']) for rec in recommendations]
        print(f"📋 [RECOMMENDATIONS] Récupération des noms pour {len(author_ids)} auteurs...")
        
        # Récupérer les noms depuis Neo4j
        author_names = {}
        try:
            author_names = neo4j_client.get_author_names(author_ids)
            print(f"✅ [RECOMMENDATIONS] {len(author_names)} noms récupérés de Neo4j")
        except Exception as name_error:
            print(f"❌ [RECOMMENDATIONS] Erreur récupération noms: {name_error}")
            # Fallback
            author_names = {str(rec['researcher_id']): f"Researcher {rec['researcher_id']}" 
                           for rec in recommendations}
        
        # Récupérer aussi le nom de l'auteur source
        source_name = "Unknown"
        try:
            source_names = neo4j_client.get_author_names([author_id])
            if author_id in source_names:
                source_name = source_names[author_id]
        except:
            pass
        
        # Enrichir les recommandations avec les noms
        enriched_recommendations = []
        for rec in recommendations:
            rec_id = str(rec['researcher_id'])
            enriched_rec = rec.copy()
            enriched_rec['name'] = author_names.get(rec_id, f"Researcher {rec_id}")
            enriched_recommendations.append(enriched_rec)
        
        ml_scores = [r['ml_score'] for r in recommendations if 'ml_score' in r]
        
        return {
            "author_id": author_id,
            "author_name": source_name,
            "strategy": strategy,
            "filter_hubs_applied": filter_hubs,
            "total_recommendations": len(recommendations),
            "score_stats": {
                "min_ml_score": min(ml_scores) if ml_scores else 0,
                "max_ml_score": max(ml_scores) if ml_scores else 0,
                "avg_ml_score": sum(ml_scores) / len(ml_scores) if ml_scores else 0
            },
            "recommendations": enriched_recommendations
        }
    except ValueError as e:
        print(f"❌ [RECOMMENDATIONS] ValueError: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"❌ [RECOMMENDATIONS] Exception: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")


# ============= AJOUTER ENDPOINT DE TEST =============
@app.get("/debug/data-coverage")
def debug_data_coverage():
    """Vérifier la couverture des données"""
    total_authors_in_neo4j = 25000  # À vérifier
    total_authors_in_model = len(analytics.node_to_idx)
    
    return {
        "authors_in_neo4j": total_authors_in_neo4j,
        "authors_in_ml_model": total_authors_in_model,
        "coverage_percentage": f"{(total_authors_in_model/total_authors_in_neo4j*100):.1f}%",
        "sample_authors_in_model": list(analytics.node_to_idx.keys())[:10],
        "has_edges": analytics.G_nx.number_of_edges() > 0,
        "communities_detected": len(set(analytics.partition.values())) if analytics.partition else 0
    }

@app.get("/debug/neo4j-test")
def debug_neo4j_test():
    """Debug: tester la connexion Neo4j et la récupération des noms"""
    try:
        # Tester la connexion
        connection_ok = neo4j_client.test_connection()
        
        # Tester avec des IDs spécifiques
        test_ids = ["3308557", "34917892", "2465270", "145845739", "3462562"]
        
        print(f"\n🔍 Test récupération noms pour {len(test_ids)} IDs...")
        names = neo4j_client.get_author_names(test_ids)
        
        return {
            "neo4j_connection": "OK" if connection_ok else "FAILED",
            "test_ids": test_ids,
            "names_retrieved": names,
            "summary": f"{len(names)}/{len(test_ids)} noms récupérés"
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "neo4j_connection": "ERROR"
        }

@app.get("/debug/model-stats")
def get_model_stats():
    """Obtenir des statistiques détaillées sur le modèle"""
    if not analytics.node_to_idx:
        return {"error": "Modèle non chargé"}
    
    # Calculer quelques métriques
    degrees = [analytics.G_nx.degree(node) for node in analytics.G_nx.nodes()]
    
    return {
        "model_info": {
            "total_authors": len(analytics.node_to_idx),
            "embeddings_shape": list(analytics.embeddings.shape),
            "device": str(analytics.device)
        },
        "graph_info": {
            "nodes": analytics.G_nx.number_of_nodes(),
            "edges": analytics.G_nx.number_of_edges(),
            "density": nx.density(analytics.G_nx),
            "isolated_nodes": len(list(nx.isolates(analytics.G_nx)))
        },
        "communities_info": {
            "total_communities": len(set(analytics.partition.values())),
            "community_sizes": {
                "min": min([list(analytics.partition.values()).count(c) for c in set(analytics.partition.values())]),
                "max": max([list(analytics.partition.values()).count(c) for c in set(analytics.partition.values())]),
                "avg": len(analytics.partition) / len(set(analytics.partition.values()))
            }
        },
        "degree_stats": {
            "min": min(degrees) if degrees else 0,
            "max": max(degrees) if degrees else 0,
            "avg": sum(degrees) / len(degrees) if degrees else 0,
            "median": np.median(degrees) if degrees else 0
        }
    }

@app.get("/debug/check-author/{author_id}")
def check_author_in_model(author_id: str):
    """Vérifier si un auteur est dans le modèle"""
    in_model = author_id in analytics.node_to_idx
    in_graph = analytics.G_nx.has_node(author_id) if analytics.G_nx else False
    
    response = {
        "author_id": author_id,
        "in_model": in_model,
        "in_graph": in_graph,
        "has_embeddings": False,
        "degree": 0,
        "community": -1
    }
    
    if in_model:
        response["has_embeddings"] = True
        if in_graph:
            response["degree"] = analytics.G_nx.degree(author_id)
            response["community"] = analytics.partition.get(author_id, -1)
            response["neighbors_count"] = len(list(analytics.G_nx.neighbors(author_id)))
    
    return response

@app.get("/debug/sample-recommendations")
def get_sample_recommendations():
    """Obtenir des recommandations pour quelques auteurs tests"""
    test_authors = ["3308557", "34917892", "2465270", "145845739", "3462562"]
    
    results = {}
    for author_id in test_authors:
        try:
            if author_id in analytics.node_to_idx:
                recs = analytics.get_recommendations(author_id, top_k=3)
                results[author_id] = {
                    "status": "success",
                    "in_model": True,
                    "recommendations_count": len(recs),
                    "recommendations": [
                        {
                            "researcher_id": r["researcher_id"],
                            "collaboration_score": r["collaboration_score"],
                            "degree": r["degree"],
                            "same_community": r["same_community"]
                        }
                        for r in recs[:2]  # Montrer seulement 2
                    ]
                }
            else:
                results[author_id] = {
                    "status": "not_in_model",
                    "in_model": False
                }
        except Exception as e:
            results[author_id] = {
                "status": "error",
                "error": str(e)
            }
    
    return {
        "test_results": results,
        "total_authors_tested": len(test_authors),
        "authors_in_model": sum(1 for r in results.values() if r.get("in_model", False))
    }

@app.get("/debug/test-diversity")
def test_diversity():
    """Test rapide : même auteur recommande-t-il des chercheurs différents ?"""
    import random
    
    # Prendre 3 auteurs aléatoires
    test_authors = random.sample(list(analytics.node_to_idx.keys()), min(3, len(analytics.node_to_idx)))
    
    results = {}
    all_recommendations = set()
    
    for author_id in test_authors:
        author_id = str(author_id)
        
        # Sans filtre
        recs_no_filter = analytics.get_recommendations(author_id, top_k=5, filter_hubs=False)
        
        # Avec filtre
        recs_with_filter = analytics.get_recommendations(author_id, top_k=5, filter_hubs=True)
        
        results[author_id] = {
            'source_degree': analytics.G_nx.degree(author_id),
            'without_filter': [r['researcher_id'] for r in recs_no_filter],
            'with_filter': [r['researcher_id'] for r in recs_with_filter],
            'overlap': len(set(r['researcher_id'] for r in recs_no_filter) & 
                          set(r['researcher_id'] for r in recs_with_filter))
        }
        
        all_recommendations.update(r['researcher_id'] for r in recs_no_filter)
    
    return {
        'test_authors': test_authors,
        'results': results,
        'analysis': {
            'total_unique_recommendations': len(all_recommendations),
            'problem_detected': len(all_recommendations) < 5,
            'diagnosis': (
                '⚠️ PROBLÈME : Moins de 5 recommandations uniques sur 3 auteurs'
                if len(all_recommendations) < 5
                else '✅ Bonne diversité des recommandations'
            )
        }
    }
@app.get("/recommendations/{author_id}/compare")
def compare_strategies(
    author_id: str = Path(..., description="ID de l'auteur"),
    top_k: int = Query(10, ge=1, le=20)
):
    """Comparer les 3 stratégies côte à côte"""
    try:
        ml_only = analytics.get_recommendations(author_id, top_k, "ml_only")
        hybrid = analytics.get_recommendations(author_id, top_k, "hybrid")
        diverse = analytics.get_recommendations(author_id, top_k, "diverse")
        
        # Analyser les différences
        ml_only_ids = set(r['researcher_id'] for r in ml_only)
        hybrid_ids = set(r['researcher_id'] for r in hybrid)
        diverse_ids = set(r['researcher_id'] for r in diverse)
        
        return {
            "author_id": author_id,
            "strategies": {
                "ml_only": ml_only,
                "hybrid": hybrid,
                "diverse": diverse
            },
            "overlap_analysis": {
                "ml_only_vs_hybrid": len(ml_only_ids & hybrid_ids),
                "ml_only_vs_diverse": len(ml_only_ids & diverse_ids),
                "hybrid_vs_diverse": len(hybrid_ids & diverse_ids),
                "unique_to_ml_only": len(ml_only_ids - hybrid_ids - diverse_ids),
                "unique_to_hybrid": len(hybrid_ids - ml_only_ids - diverse_ids),
                "unique_to_diverse": len(diverse_ids - ml_only_ids - hybrid_ids)
            }
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


@app.get("/recommendations/by-name/{author_name}")
def get_recommendations_by_name(
    author_name: str,
    top_k: int = Query(10, ge=1, le=50)
):
    """Trouver des recommandations par nom d'auteur"""
    
    try:
        # 1. Chercher l'auteur dans le graphe par nom
        matching_nodes = []
        for node_id in analytics.G_full.nodes():
            node_name = analytics.G_full.nodes[node_id].get('name', '')
            if author_name.lower() in node_name.lower():
                matching_nodes.append(node_id)
        
        if not matching_nodes:
            raise HTTPException(status_code=404, detail=f"Auteur '{author_name}' non trouvé")
        
        # Prendre le premier match
        model_id = matching_nodes[0]
        author_name_found = analytics.G_full.nodes[model_id].get('name', 'inconnu')
        
        # 2. Vérifier que l'auteur est dans les structures nécessaires
        if model_id not in analytics.node_to_idx:
            raise HTTPException(
                status_code=404, 
                detail=f"Auteur trouvé ({author_name_found}) mais pas dans le modèle ML"
            )
        
        # 3. Vérifier le cache
        if model_id not in analytics.metrics_cache:
            # Calculer les métriques manquantes
            analytics._compute_missing_metrics([model_id])
        
        # 4. Récupérer les recommandations
        recommendations = analytics.get_recommendations(str(model_id), top_k)
        
        # 5. Enrichir avec les noms et vérifier chaque recommandation
        enriched_recs = []
        for rec in recommendations:
            rec_id = rec['researcher_id']
            
            # Vérifier si la recommandation est dans le cache
            if rec_id not in analytics.metrics_cache:
                analytics._compute_missing_metrics([rec_id])
            
            # Récupérer le nom depuis le graphe
            rec_name = analytics.G_full.nodes[rec_id].get('name', f"Author {rec_id}")
            
            enriched_rec = rec.copy()
            enriched_rec['name'] = rec_name
            enriched_recs.append(enriched_rec)
        
        return {
            "author": author_name_found,
            "model_id": str(model_id),
            "total_recommendations": len(enriched_recs),
            "recommendations": enriched_recs
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Erreur dans get_recommendations_by_name: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Erreur interne: {str(e)}"
        )  

@app.get("/analytics/author/{author_id}")
def get_author_analytics_endpoint(
    author_id: str = Path(..., description="ID de l'auteur")
):
    """Obtenir les métriques détaillées d'un auteur"""
    try:
        return analytics.get_author_analytics(author_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@app.get("/debug/check-node/{author_id}")
def check_node(author_id: str):
    """Debug: vérifier si un auteur existe"""
    try:
        author_id_int = int(author_id)
    except ValueError:
        author_id_int = None
    
    return {
        "author_id": author_id,
        "exists_in_model": (author_id in analytics.node_to_idx or 
                           (author_id_int and author_id_int in analytics.node_to_idx)),
        "exists_in_full_graph": (author_id in analytics.G_full or 
                                 (author_id_int and author_id_int in analytics.G_full)),
        "exists_in_ml_graph": (author_id in analytics.G_nx or 
                              (author_id_int and author_id_int in analytics.G_nx)) if analytics.G_nx else False,
        "total_nodes_full": analytics.G_full.number_of_nodes() if analytics.G_full else 0,
        "total_nodes_model": len(analytics.node_to_idx) if analytics.node_to_idx else 0,
        "sample_nodes": list(analytics.G_full.nodes())[:10] if analytics.G_full else []
    }
@app.get("/graph/nodes")
def get_graph_nodes():
    if analytics.node_to_idx is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé")
    
    return {
        "total_nodes": len(analytics.node_to_idx),
        "nodes": list(analytics.node_to_idx.keys())
    }

@app.get("/diagnostic/model-variance")
def check_model_variance():
    """Vérifier la variance des scores ML sans crash."""

    import random
    results = {}

    # Tirer uniquement des nœuds dans G_nx
    valid_nodes = list(analytics.G_nx.nodes())

    # Sample 10 auteurs du modèle ML
    sample_researchers = random.sample(valid_nodes, min(10, len(valid_nodes)))

    for researcher_id in sample_researchers:

        src_idx = analytics.node_to_idx[researcher_id]

        # Skip si isolé (pas de voisins)
        existing_neighbors = set(analytics.G_nx.neighbors(researcher_id))
        
        # Prendre candidats dans G_nx uniquement
        candidates = [
            node for node in analytics.G_nx.nodes()
            if node != researcher_id and node not in existing_neighbors
        ]

        if len(candidates) == 0:
            continue

        # Limiter à 100
        candidates = random.sample(candidates, min(100, len(candidates)))
        candidate_indices = [analytics.node_to_idx[node] for node in candidates]

        # Scores ML
        with torch.no_grad():
            edge_index = torch.tensor(
                [[src_idx] * len(candidate_indices), candidate_indices],
                dtype=torch.long
            ).to(analytics.device)

            scores = analytics.predictor(analytics.embeddings, edge_index).cpu().numpy()

        results[researcher_id] = {
            "min_score": float(scores.min()),
            "max_score": float(scores.max()),
            "mean_score": float(scores.mean()),
            "std_score": float(scores.std()),
            "median_score": float(np.median(scores)),
            "num_candidates": len(candidates)
        }

    # Analyse globale
    all_stds = [v["std_score"] for v in results.values()]
    all_ranges = [v["max_score"] - v["min_score"] for v in results.values()]

    return {
        "analysis": {
            "avg_std_dev": float(np.mean(all_stds)),
            "avg_score_range": float(np.mean(all_ranges)),
            "interpretation": (
                "⚠️ Scores trop similaires → modèle peu discriminant"
                if np.mean(all_stds) < 0.05 else
                "✅ Variance correcte : modèle discriminant"
            )
        },
        "per_researcher": results
    }

@app.get("/diagnostic/diversity-check")
# ============= ENDPOINTS COMMUNITÉS =============
@app.get("/api/communities")
async def get_all_communities():
    """Retourne toutes les communautés détectées - VERSION CORRIGÉE"""
    if analytics.partition is None:
        raise HTTPException(status_code=500, detail="Communities not computed")
    
    # Organiser les nœuds par communauté
    communities_dict = {}
    for node, comm_id in analytics.partition.items():
        if comm_id not in communities_dict:
            communities_dict[comm_id] = []
        communities_dict[comm_id].append(node)
    
    # FILTRE AMÉLIORÉ : Ne garder que les communautés significatives
    # 1. Taille > 5 membres
    # 2. Densité > 0.1 (ou a des arêtes internes)
    filtered_communities = {}
    for comm_id, nodes in communities_dict.items():
        if len(nodes) >= 5:  # Augmenter le seuil minimal
            subgraph = analytics.G_nx.subgraph(nodes)
            # Vérifier si la communauté a des collaborations internes
            if subgraph.number_of_edges() > 0:
                filtered_communities[comm_id] = nodes
    
    print(f"📊 Communautés : {len(communities_dict)} totales → {len(filtered_communities)} après filtrage (≥5 membres)")
    
    # Si pas assez, prendre les 50 plus grandes
    if len(filtered_communities) < 10:
        print("⚠️  Peu de communautés significatives, prise des top 50 par taille")
        sorted_comms = sorted(communities_dict.items(), key=lambda x: len(x[1]), reverse=True)[:50]
        filtered_communities = dict(sorted_comms)
    
    # Créer la structure de réponse
    communities_list = []
    
    # Récupérer tous les IDs uniques pour les top membres
    all_author_ids = []
    for nodes in list(filtered_communities.values())[:50]:  # Limiter à 50 communautés
        # Prendre les 5 premiers membres par centralité
        community_members = []
        for node in nodes:
            if node in analytics.metrics_cache:
                metrics = analytics.metrics_cache[node]
                community_members.append({
                    "node": node,
                    "centrality": metrics.get("degree_centrality", 0),
                    "degree": metrics.get("degree", 0)
                })
        
        # Trier par centralité et prendre top 5
        community_members.sort(key=lambda x: x["centrality"], reverse=True)
        top_nodes = [str(m["node"]) for m in community_members[:5]]
        all_author_ids.extend(top_nodes)
    
    # Récupérer les noms en batch
    author_names = {}
    try:
        # Prendre uniquement les IDs uniques
        unique_ids = list(set(all_author_ids))
        print(f"📋 Récupération des noms pour {len(unique_ids)} auteurs uniques...")
        author_names = neo4j_client.get_author_names(unique_ids)
    except Exception as e:
        print(f"⚠️  Impossible de récupérer les noms: {e}")
        author_names = {}
    
    # Construire la réponse
    for comm_id, nodes in list(filtered_communities.items())[:50]:  # Limiter à 50 communautés max
        subgraph = analytics.G_nx.subgraph(nodes)
        
        # Calculer les métriques
        density = nx.density(subgraph) if subgraph.number_of_edges() > 0 else 0
        
        # Trouver les top membres par centralité
        top_members_info = []
        for node in nodes:
            if node in analytics.metrics_cache:
                metrics = analytics.metrics_cache[node]
                top_members_info.append({
                    "author_id": str(node),
                    "centrality": metrics.get("degree_centrality", 0),
                    "degree": metrics.get("degree", 0),
                    "pagerank": metrics.get("pagerank", 0)
                })
        
        # Trier et garder top 5
        top_members_info.sort(key=lambda x: x["centrality"], reverse=True)
        top_members_info = top_members_info[:5]
        
        # Enrichir avec les noms
        top_members = []
        for member in top_members_info:
            author_id = member["author_id"]
            name = author_names.get(author_id, f"Author {author_id}")
            # Si le nom est générique, essayer de le trouver dans le graphe
            if name.startswith("Author "):
                try:
                    node_id = int(author_id) if author_id.isdigit() else author_id
                    if node_id in analytics.G_full.nodes():
                        graph_name = analytics.G_full.nodes[node_id].get('name')
                        if graph_name:
                            name = graph_name
                except:
                    pass
            
            top_members.append({
                "author_id": author_id,
                "name": name,
                "publication_count": member["degree"],
                "centrality_score": member["centrality"]
            })
        
        # Générer un nom intelligent
        community_name = generate_community_name_from_members(top_members, comm_id)
        
        # Détecter les topics
        topics = detect_topics_from_names(top_members)
        
        communities_list.append({
            "id": comm_id,
            "name": community_name,
            "size": len(nodes),
            "density": round(density, 3),
            "internal_edges": subgraph.number_of_edges(),
            "topics": topics[:3],  # Max 3 topics
            "top_members": top_members,
            "description": f"Communauté de {len(nodes)} chercheurs spécialisés en {', '.join(topics[:2])}"
        })
    
    # Trier par taille décroissante
    communities_list.sort(key=lambda x: x["size"], reverse=True)
    
    return {
        "communities": communities_list[:30],  # Retourner max 30 communautés
        "total_communities": len(communities_list),
        "total_authors": sum(comm["size"] for comm in communities_list[:30]),
        "stats": {
            "filtered_out": len(communities_dict) - len(communities_list),
            "average_size": sum(comm["size"] for comm in communities_list[:30]) / len(communities_list[:30]) if communities_list else 0
        }
    }

def generate_community_name_from_members(top_members, community_id):
    """Génère un nom significatif à partir des membres"""
    if not top_members:
        return f"Research Group {community_id}"
    
    # Extraire les noms
    names = [m["name"] for m in top_members[:3]]
    
    # Chercher des mots-clés dans les noms
    domain_keywords = {
        "AI Research": ["hinton", "bengio", "lecun", "hochreiter", "schmidhuber"],
        "Deep Learning": ["deep", "neural", "network", "learning"],
        "Computer Vision": ["vision", "image", "visual", "recognition"],
        "NLP": ["language", "linguistic", "translation", "text"],
        "Physics": ["physic", "quantum", "particle", "astrophysic"],
        "Mathematics": ["math", "statistic", "algebra", "calculus"],
        "Biology": ["bio", "genetic", "cell", "molecular"],
        "Medical": ["medical", "health", "clinical", "hospital"]
    }
    
    for domain, keywords in domain_keywords.items():
        for name in names:
            if any(keyword in name.lower() for keyword in keywords):
                return f"{domain} Network"
    
    # Utiliser le nom du premier auteur
    first_name = names[0]
    # Extraire le nom de famille (dernier mot)
    last_name = first_name.split()[-1] if " " in first_name else first_name
    return f"{last_name} Research Group"

def detect_topics_from_names(top_members):
    """Détecte les domaines à partir des noms des membres"""
    topics = []
    name_keywords = {
        "Computer Science": ["computer", "cs", "ai", "machine", "learning", "software", "algorithm"],
        "Physics": ["physic", "quantum", "particle", "astrophysic", "cosmology"],
        "Mathematics": ["math", "statistic", "algebra", "calculus", "geometry"],
        "Biology": ["bio", "genetic", "cell", "molecular", "ecology"],
        "Medicine": ["medical", "health", "clinical", "pharmacy", "surgery"],
        "Engineering": ["engineer", "mechanical", "electrical", "civil"],
        "Chemistry": ["chem", "molecule", "organic", "inorganic"]
    }
    
    for member in top_members[:5]:  # Regarder les 5 premiers
        name_lower = member["name"].lower()
        for topic, keywords in name_keywords.items():
            if any(keyword in name_lower for keyword in keywords):
                if topic not in topics:
                    topics.append(topic)
    
    # Valeurs par défaut si rien trouvé
    if not topics:
        topics = ["Computer Science", "Mathematics"]
    
    return topics[:3]  # Maximum 3 topics

def generate_meaningful_community_name(community_id, top_members):
    """Génère un nom significatif pour la communauté"""
    if not top_members:
        return f"Research Group {community_id}"
    
    # Utiliser les noms des top membres
    top_names = [m["name"] for m in top_members[:3]]
    
    # Chercher des mots-clés dans les noms
    keywords = {
        "AI": ["hinton", "bengio", "lecun", "hochreiter", "schmidhuber"],
        "Deep Learning": ["deep", "learning", "neural", "network"],
        "Computer Vision": ["vision", "image", "visual"],
        "NLP": ["language", "linguistic", "text", "nlp"],
        "Physics": ["physic", "quantum", "particle"],
        "Mathematics": ["math", "statistic", "algebra"],
        "Biology": ["bio", "genetic", "cell", "dna"],
        "Medical": ["medical", "health", "clinical", "hospital"]
    }
    
    for domain, words in keywords.items():
        for name in top_names:
            if any(word.lower() in name.lower() for word in words):
                return f"{domain} Research Network"
    
    # Sinon utiliser le nom du premier auteur
    first_name = top_members[0]["name"].split()[-1]  # Dernier nom
    return f"{first_name} Research Group"

def detect_community_topics(top_members):
    """Détecte les domaines de recherche de la communauté"""
    topics = ["Computer Science", "Mathematics"]  # Par défaut
    
    # Analyser les noms des membres
    for member in top_members[:5]:
        name_lower = member["name"].lower()
        
        if any(word in name_lower for word in ["ai", "learning", "neural"]):
            if "Artificial Intelligence" not in topics:
                topics.append("Artificial Intelligence")
        elif any(word in name_lower for word in ["physic", "quantum"]):
            if "Physics" not in topics:
                topics.append("Physics")
        elif any(word in name_lower for word in ["bio", "genetic"]):
            if "Biology" not in topics:
                topics.append("Biology")
        elif any(word in name_lower for word in ["medical", "health"]):
            if "Medicine" not in topics:
                topics.append("Medicine")
    
    return topics[:3]  # Retourner max 3 topics

@app.get("/api/communities/summary")
async def get_communities_summary():
    """Retourne un résumé des communautés"""
    if analytics.partition is None:
        raise HTTPException(status_code=500, detail="Communities not computed")
    
    communities = await get_all_communities()
    communities_data = communities.get("communities", [])
    
    if not communities_data:
        return {
            "total_communities": 0,
            "total_authors": 0,
            "average_size": 0,
            "average_density": 0
        }
    
    return {
        "total_communities": len(communities_data),
        "total_authors": sum(c["size"] for c in communities_data),
        "average_size": sum(c["size"] for c in communities_data) / len(communities_data),
        "average_density": sum(c["density"] for c in communities_data) / len(communities_data),
        "largest_community": {
            "id": communities_data[0]["id"],
            "name": communities_data[0]["name"],
            "size": communities_data[0]["size"]
        },
        "description": "Communities detected using Louvain algorithm with modularity optimization"
    }

@app.get("/api/communities/{community_id}")
async def get_community(community_id: int):
    """Retourne les détails d'une communauté spécifique AVEC VRAIS NOMS"""
    if analytics.partition is None:
        raise HTTPException(status_code=500, detail="Communities not computed")
    
    # Trouver tous les nœuds de cette communauté
    community_nodes = [
        node for node, comm in analytics.partition.items()
        if comm == community_id
    ]
    
    if not community_nodes:
        raise HTTPException(status_code=404, detail=f"Community {community_id} not found")
    
    # Créer le sous-graphe
    subgraph = analytics.G_nx.subgraph(community_nodes)
    
    # Trouver les top membres
    top_members = []
    for node in community_nodes:
        if node in analytics.metrics_cache:
            metrics = analytics.metrics_cache[node]
            top_members.append({
                "author_id": str(node),
                "name": f"Author {node}",  # Temporaire, sera remplacé
                "publication_count": metrics.get("degree", 0),
                "centrality_score": metrics.get("degree_centrality", 0),
                "pagerank": metrics.get("pagerank", 0)
            })
    
    # Trier par centralité
    top_members.sort(key=lambda x: x["centrality_score"], reverse=True)
    
    # Récupérer les vrais noms depuis Neo4j
    author_ids = [member["author_id"] for member in top_members[:20]]
    try:
        author_names = neo4j_client.get_author_names(author_ids)
        
        # Remplacer les noms temporaires par les vrais noms
        for member in top_members[:20]:
            author_id = member["author_id"]
            if author_id in author_names:
                member["name"] = author_names[author_id]
    except Exception as e:
        print(f"⚠️ Erreur récupération noms pour communauté {community_id}: {e}")
        # Fallback: chercher dans le graphe
        for member in top_members[:20]:
            node_id = member["author_id"]
            try:
                # Convertir en int si possible
                try:
                    node_int = int(node_id)
                    if node_int in analytics.G_full.nodes():
                        graph_name = analytics.G_full.nodes[node_int].get('name')
                        if graph_name:
                            member["name"] = graph_name
                except:
                    if node_id in analytics.G_full.nodes():
                        graph_name = analytics.G_full.nodes[node_id].get('name')
                        if graph_name:
                            member["name"] = graph_name
            except:
                pass
    
    # Calculer les métriques
    degree_sequence = [subgraph.degree(node) for node in community_nodes]
    
    # Générer un nom significatif pour la communauté
    community_name = generate_meaningful_community_name_from_members(
        top_members[:5], community_id
    )
    
    return {
        "id": community_id,
        "name": community_name,
        "size": len(community_nodes),
        "density": round(nx.density(subgraph), 3),
        "internal_edges": subgraph.number_of_edges(),
        "average_degree": sum(degree_sequence) / len(degree_sequence),
        "top_members": top_members[:20],  # Top 20 membres
        "metrics": {
            "min_degree": min(degree_sequence),
            "max_degree": max(degree_sequence),
            "avg_degree": sum(degree_sequence) / len(degree_sequence),
            "diameter": nx.diameter(subgraph) if nx.is_connected(subgraph) else None,
            "avg_clustering": nx.average_clustering(subgraph)
        },
        "description": f"Communauté de {len(community_nodes)} chercheurs spécialisés en collaboration"
    }

def generate_meaningful_community_name_from_members(top_members, community_id):
    """Génère un nom significatif à partir des membres de la communauté"""
    if not top_members or len(top_members) == 0:
        return f"Research Community {community_id}"
    
    # Utiliser les vrais noms des top membres
    top_names = [member["name"] for member in top_members[:3] if "name" in member]
    
    # Chercher des mots-clés dans les noms
    keywords = {
        "AI Research": ["hinton", "bengio", "lecun", "hochreiter", "schmidhuber"],
        "Deep Learning": ["clevert", "unterthiner", "krizhevsky", "sutskever"],
        "Computer Vision": ["litjens", "corrado", "bae", "vision", "image"],
        "Medical AI": ["medical", "health", "clinical", "hospital", "diagnostic"],
        "Physics": ["physic", "quantum", "particle", "astrophysic"],
        "Mathematics": ["math", "statistic", "algebra", "calculus"]
    }
    
    for domain, words in keywords.items():
        for name in top_names:
            if any(word.lower() in name.lower() for word in words):
                return f"{domain} Network"
    
    # Sinon utiliser le nom du premier auteur
    if top_names:
        first_name = top_names[0]
        # Extraire le nom de famille (dernier mot)
        last_name = first_name.split()[-1] if " " in first_name else first_name
        return f"{last_name} Research Group"
    
    return f"Research Community {community_id}"

@app.get("/api/communities/author/{author_id}")
async def get_author_community(author_id: str):
    """Retourne la communauté d'un auteur spécifique"""
    if analytics.partition is None:
        raise HTTPException(status_code=500, detail="Communities not computed")
    
    try:
        # Essayer de convertir en int
        author_id_int = int(author_id)
        author_key = author_id_int if author_id_int in analytics.partition else author_id
    except ValueError:
        author_key = author_id
    
    if author_key not in analytics.partition:
        raise HTTPException(status_code=404, detail=f"Author {author_id} not found in communities")
    
    community_id = analytics.partition[author_key]
    
    # Récupérer les détails de la communauté
    community_details = await get_community(community_id)
    
    return {
        "author": {
            "author_id": str(author_id),
            "name": f"Author {author_id}",
            "community_id": community_id
        },
        "community": community_details
    }

@app.get("/api/communities/largest/{n}")
async def get_largest_communities(n: int):
    """Retourne les n plus grandes communautés"""
    if analytics.partition is None:
        raise HTTPException(status_code=500, detail="Communities not computed")
    
    # Organiser les nœuds par communauté
    communities_dict = {}
    for node, comm_id in analytics.partition.items():
        if comm_id not in communities_dict:
            communities_dict[comm_id] = []
        communities_dict[comm_id].append(node)
    
    # Trier par taille
    sorted_communities = sorted(
        communities_dict.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )[:n]
    
    communities_list = []
    for comm_id, nodes in sorted_communities:
        communities_list.append({
            "id": comm_id,
            "name": f"Research Community {comm_id}",
            "size": len(nodes)
        })
    
    return {
        "count": len(communities_list),
        "communities": communities_list
    }
def check_recommendation_diversity():
    """Vérifier si différents auteurs reçoivent des recommandations différentes"""
    import random
    
    # Sélectionner 5 auteurs aléatoires
    valid_authors = [
        a for a in list(analytics.G_nx.nodes())
        if analytics.G_nx.degree(a) > 0
    ]
    
    test_authors = random.sample(valid_authors, min(5, len(valid_authors)))
    
    all_recommendations = []
    results_per_author = {}
    
    for author in test_authors:
        recs = analytics.get_recommendations(str(author), top_k=10, filter_hubs=True)
        rec_ids = [r['researcher_id'] for r in recs]
        
        all_recommendations.extend(rec_ids)
        
        results_per_author[str(author)] = {
            'source_degree': analytics.G_nx.degree(author),
            'source_community': analytics.partition.get(author, -1),
            'recommendations': rec_ids[:5],  # Top 5 seulement
            'avg_rec_degree': round(
                sum(r['degree'] for r in recs) / len(recs),
                1
            ),
            'unique_communities': len(set(r['community_id'] for r in recs))
        }
    
    # Analyse globale
    from collections import Counter
    rec_counts = Counter(all_recommendations)
    most_common = rec_counts.most_common(10)
    
    total_unique = len(set(all_recommendations))
    total_recs = len(all_recommendations)
    
    # Diagnostics
    is_diverse = total_unique > (total_recs * 0.7)  # Au moins 70% uniques
    
    return {
        "test_authors": len(test_authors),
        "total_recommendations_given": total_recs,
        "unique_researchers_recommended": total_unique,
        "diversity_ratio": round(total_unique / total_recs, 3),
        "status": "✅ DIVERSITÉ OK" if is_diverse else "⚠️ PROBLÈME DE DIVERSITÉ",
        "most_recommended": convert_to_python_types([
            {
                "researcher_id": rec_id,
                "times_recommended": count,
                "degree": analytics.G_nx.degree(rec_id),
                "is_hub": analytics.G_nx.degree(rec_id) > np.percentile(
                    [analytics.G_nx.degree(n) for n in analytics.G_nx.nodes()],
                    95
                )
            }
            for rec_id, count in most_common
        ]),
        "per_author_analysis": convert_to_python_types(results_per_author)
    }

@app.get("/debug/hub-analysis")
def analyze_hubs():
    """Analyser si quelques hubs dominent toutes les recommandations"""
    
    # Trouver les top hubs par degré
    degrees = dict(analytics.G_nx.degree())
    top_hubs = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:50]
    
    # Trouver les top hubs par PageRank
    pageranks = [(node, metrics['pagerank']) for node, metrics in analytics.metrics_cache.items()]
    top_pr = sorted(pageranks, key=lambda x: x[1], reverse=True)[:50]
    
    # Analyser les embeddings des hubs
    hub_ids = [h[0] for h in top_hubs[:10]]
    hub_indices = [analytics.node_to_idx[h] for h in hub_ids if h in analytics.node_to_idx]
    
    if len(hub_indices) > 0:
        hub_embeddings = analytics.embeddings[hub_indices].cpu().numpy()
        
        # Similarité entre hubs
        from sklearn.metrics.pairwise import cosine_similarity
        hub_similarities = cosine_similarity(hub_embeddings)
        
        # Moyenne de similarité (hors diagonale)
        mask = np.ones(hub_similarities.shape, dtype=bool)
        np.fill_diagonal(mask, False)
        avg_hub_similarity = float(np.mean(hub_similarities[mask]))
    else:
        avg_hub_similarity = None
    
    # Prédire leur score avec un auteur aléatoire
    import random
    test_author = random.choice(list(analytics.node_to_idx.keys()))
    test_idx = analytics.node_to_idx[test_author]
    
    hub_scores = []
    for hub_id in hub_ids[:10]:
        if hub_id in analytics.node_to_idx and hub_id != test_author:
            hub_idx = analytics.node_to_idx[hub_id]
            
            with torch.no_grad():
                edge_index = torch.tensor([[test_idx], [hub_idx]], dtype=torch.long).to(analytics.device)
                score = analytics.predictor(analytics.embeddings, edge_index).cpu().item()
            
            hub_scores.append({
                'hub_id': hub_id,
                'degree': degrees[hub_id],
                'pagerank': analytics.metrics_cache.get(hub_id, {}).get('pagerank', 0),
                'ml_score_with_test_author': score
            })
    
    return {
        "top_10_hubs_by_degree": [
            {
                'node_id': node,
                'degree': degree,
                'pagerank': analytics.metrics_cache.get(node, {}).get('pagerank', 0)
            }
            for node, degree in top_hubs[:10]
        ],
        "top_10_by_pagerank": [
            {
                'node_id': node,
                'pagerank': pr,
                'degree': degrees.get(node, 0)
            }
            for node, pr in top_pr[:10]
        ],
        "hub_embedding_similarity": avg_hub_similarity,
        "hub_ml_scores": hub_scores,
        "diagnosis": {
            "high_hub_similarity": avg_hub_similarity is not None and avg_hub_similarity > 0.9,
            "hubs_get_high_scores": any(s['ml_score_with_test_author'] > 0.8 for s in hub_scores),
            "message": (
                "⚠️ PROBLÈME : Les hubs ont des embeddings très similaires ET obtiennent des scores élevés"
                if (avg_hub_similarity is not None and avg_hub_similarity > 0.9 and 
                    any(s['ml_score_with_test_author'] > 0.8 for s in hub_scores))
                else "✅ Les hubs ne dominent pas anormalement"
            )
        }
    }

@app.get("/communities/{community_id}/graph")
def get_community_graph(community_id: int):
    if analytics.partition is None:
        raise HTTPException(status_code=500, detail="Communities not computed")

    community_nodes = [
        node for node, comm in analytics.partition.items()
        if comm == community_id
    ]

    if not community_nodes:
        return {
            "community_id": community_id,
            "nodes": [],
            "edges": []
        }

    # Extraire sous-graphe
    subgraph = analytics.G_full.subgraph(community_nodes).copy()

    nodes = [{"id": str(node), "degree": subgraph.degree(node)} for node in subgraph.nodes()]
    edges = [{"source": str(u), "target": str(v)} for u, v in subgraph.edges()]

    return {
        "community_id": community_id,
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "nodes": nodes,
        "edges": edges
    }



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)