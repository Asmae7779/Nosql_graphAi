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

# ============= CLASSES MODÈLE =============
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
        self.partition = None
        self.metrics_cache = {}
        
    def load_model(self, model_path: str, config_path: str):
       """Charger le modèle et calculer les métriques"""
       print(f"🔄 Chargement du modèle depuis {model_path}...")
       
       # Charger config
       with open(config_path, 'r') as f:
           config = yaml.safe_load(f)
       
       # Charger checkpoint
       checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
       
       # Vérifier format du checkpoint
       if 'embeddings' in checkpoint:
           # ========== FORMAT API COMPLET ==========
           print("✅ Format API complet détecté")
           self.embeddings = checkpoint['embeddings'].to(self.device)
           self.node_to_idx = checkpoint['node_to_idx']
           self.idx_to_node = {v: k for k, v in self.node_to_idx.items()}
           
           embedding_dim = self.embeddings.shape[1]
           
           self.encoder = GraphSAGE(3, 128, embedding_dim).to(self.device)
           self.predictor = EdgePredictor(embedding_dim).to(self.device)
           
           self.encoder.load_state_dict(checkpoint['model_state']['encoder'])
           self.predictor.load_state_dict(checkpoint['model_state']['predictor'])
           
           self.encoder.eval()
           self.predictor.eval()
           
           # Charger graphe
           cache_path = "../data/graph_lcc.pkl"
           print(f"📥 Chargement du graphe depuis {cache_path}...")
           with open(cache_path, 'rb') as f:
               self.G_nx = pickle.load(f)
           
           print(f"✅ Graphe G_nx chargé : {self.G_nx.number_of_nodes()} nœuds")
           
       else:
           
           
           # Charger graphe AVANT tout
           cache_path = "../data/graph_lcc.pkl"
           print(f"📥 Chargement du graphe depuis {cache_path}...")
           
           if not os.path.exists(cache_path):
               raise FileNotFoundError(f"Graphe non trouvé : {cache_path}")
           
           with open(cache_path, 'rb') as f:
               self.G_nx = pickle.load(f)
           
           print(f"✅ Graphe G_nx chargé : {self.G_nx.number_of_nodes()} nœuds, {self.G_nx.number_of_edges()} arêtes")
           
           # Créer node_to_idx depuis le graphe
           node_list = list(self.G_nx.nodes())
           self.node_to_idx = {node: idx for idx, node in enumerate(node_list)}
           self.idx_to_node = {v: k for k, v in self.node_to_idx.items()}
           
           print(f"✅ node_to_idx créé : {len(self.node_to_idx)} nœuds")
           
           # Calculer features
           print("📊 Calcul des features...")
           deg = dict(self.G_nx.degree())
           pagerank = nx.pagerank(self.G_nx, max_iter=50)
           clustering = nx.clustering(self.G_nx)
           
           x = torch.tensor([
               [deg[node], pagerank[node], clustering[node]] 
               for node in node_list
           ], dtype=torch.float).to(self.device)
           
           # Créer edge_index
           edges = [(self.node_to_idx[u], self.node_to_idx[v]) for u, v in self.G_nx.edges()]
           edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(self.device)
           
           print(f"✅ Features calculées : {x.shape}")
           print(f"✅ Arêtes : {edge_index.shape}")
           
           # Charger encoder et générer embeddings
           embedding_dim = 64
           self.encoder = GraphSAGE(3, 128, embedding_dim).to(self.device)
           self.encoder.load_state_dict(checkpoint['encoder'])
           self.encoder.eval()
           
           with torch.no_grad():
               self.embeddings = self.encoder(x, edge_index)
           
           print(f"✅ Embeddings générés : {self.embeddings.shape}")
           
           # Charger predictor
           self.predictor = EdgePredictor(embedding_dim).to(self.device)
           self.predictor.load_state_dict(checkpoint['predictor'])
           self.predictor.eval()
       
       # ========== COMMUN AUX DEUX FORMATS ==========
       
       # ✅ CRITIQUE : Définir G_full = G_nx
       self.G_full = self.G_nx
       print(f"✅ G_full défini : {self.G_full.number_of_nodes()} nœuds")
       
       # Charger communautés
       partition_path = "../results/communities/louvain_partition.pkl"
       try:
           with open(partition_path, 'rb') as f:
               self.partition = pickle.load(f)
           print(f"✅ Communautés chargées : {len(set(self.partition.values()))} communautés")
       except Exception as e:
           print(f"⚠️ Calcul des communautés : {e}")
           import community as community_louvain
           self.partition = community_louvain.best_partition(self.G_nx)
           print(f"✅ Communautés calculées : {len(set(self.partition.values()))} communautés")
       
       # Précalculer métriques
       print("📊 Calcul des métriques globales...")
       self._compute_global_metrics()
       
       # ✅ Affichage final de vérification
       print(f"\n{'='*60}")
       print(f"✅ MODÈLE CHARGÉ AVEC SUCCÈS")
       print(f"{'='*60}")
       print(f"Device                : {self.device}")
       print(f"G_nx (graphe ML)      : {self.G_nx.number_of_nodes()} nœuds, {self.G_nx.number_of_edges()} arêtes")
       print(f"G_full (graphe full)  : {self.G_full.number_of_nodes() if self.G_full else 0} nœuds")
       print(f"node_to_idx           : {len(self.node_to_idx)} entrées")
       print(f"Embeddings            : {self.embeddings.shape}")
       print(f"Communautés           : {len(set(self.partition.values()))}")
       print(f"Métriques en cache    : {len(self.metrics_cache)}")
       print(f"{'='*60}\n")

    def _load_complete_graph_from_neo4j(self, config):
      """Charger TOUS les auteurs depuis Neo4j"""
      from neo4j import GraphDatabase
      
      neo4j_config = config['neo4j']
      
      driver = GraphDatabase.driver(
          neo4j_config['uri'],
          auth=(neo4j_config['user'], neo4j_config['password'])
      )
      
      G = nx.Graph()
      
      print("📥 Connexion à Neo4j...")
      
      with driver.session() as session:
          # ✅ Charger tous les auteurs (en filtrant les None)
          print("📥 Chargement des auteurs...")
          result = session.run("""
              MATCH (a:Author) 
              WHERE a.authorId IS NOT NULL
              RETURN a.authordId AS id
          """)
          
          author_count = 0
          skipped = 0
          for record in result:
              author_id = record["id"]
              if author_id is not None:  # Double vérification
                  G.add_node(author_id)
                  author_count += 1
              else:
                  skipped += 1
          
          print(f"✅ {author_count} auteurs chargés")
          if skipped > 0:
              print(f"⚠️  {skipped} auteurs ignorés (id=NULL)")
          
          # ✅ Charger toutes les collaborations (en filtrant les None)
          print("📥 Chargement des collaborations...")
          result = session.run("""
             MATCH (a1:authors)-[:AUTHORED]->(p:papers)<-[:AUTHORED]-(a2:authors)
             WHERE a1.authorId IS NOT NULL AND a2.authorId IS NOT NULL AND a1.authorId < a2.authorId
             RETURN a1.authorId AS src, a2.authorId AS dst
          """)
          
          edge_count = 0
          edge_skipped = 0
          for record in result:
              src = record["src"]
              dst = record["dst"]
              
              # Vérifier que les IDs sont valides
              if src is not None and dst is not None:
                  if G.has_node(src) and G.has_node(dst):
                      G.add_edge(src, dst)
                      edge_count += 1
                  else:
                      edge_skipped += 1
              else:
                  edge_skipped += 1
          
          print(f"✅ {edge_count} collaborations chargées")
          if edge_skipped > 0:
              print(f"⚠️  {edge_skipped} collaborations ignorées (nœuds manquants)")
      
      driver.close()
      
      # Statistiques
      isolated = list(nx.isolates(G))
      print(f"ℹ️  {len(isolated)} auteurs isolés (sans collaborations)")
      
      return G
      
    def _compute_global_metrics(self):
        """Calculer métriques pour tous les nœuds (avec cache sur disque)"""
        import os
        
        # Chemin du cache
        cache_file = "../results/metrics_cache.pkl"
        
        # Essayer de charger depuis le cache
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
        
        print(f"✅ Métriques calculées et sauvegardées")

    
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
  

    def _normalize_id(self, author_id):
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
      
      researcher_id = str(researcher_id)
  
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
          pr = self.metrics_cache[candidate]["pagerank"]
          comm = self.partition.get(candidate, -1)
          
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
              "degree": int(deg),  # ✅ Convertir en int Python
              "degree_percentile": round(
                  sum(1 for d in all_degrees if d < deg) / len(all_degrees) * 100, 
                  1
              ),
              "same_community": bool(comm == src_community),  # ✅ Convertir en bool Python
              "community_id": int(comm),  # ✅ Convertir en int Python
              "pagerank": float(pr),  # ✅ Convertir en float Python
              "common_neighbors": int(common_neighbors),  # ✅ Convertir en int Python
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

# ============= INITIALISATION FASTAPI =============
analytics = ResearcherAnalytics()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charger le modèle au démarrage"""
    import os
    
    # Chemins possibles
    possible_models = [
        "../models/link_prediction_model.pkl",   # Votre chemin actuel
        "../results/link_prediction_model.pt",   # Format API standard       
    ]
    
    model_path = None
    for path in possible_models:
        if os.path.exists(path):
            model_path = path
            print(f"✅ Modèle trouvé: {path}")
            break
    
    if not model_path:
        raise FileNotFoundError(
            f"Aucun modèle trouvé. Cherché:\n" + 
            "\n".join(f"  - {p}" for p in possible_models)
        )
    
    config_path = "../config.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config non trouvée: {config_path}")
    
    analytics.load_model(model_path=model_path, config_path=config_path)
    yield

app = FastAPI(
    title="Research Collaboration API",
    description="API ML pour recommandations de collaborations",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    filter_hubs: bool = Query(False, description="Exclure hubs (top 5% degree)")  # ✅ AJOUT
):

    try:
        recommendations = analytics.get_recommendations(
            author_id, top_k, strategy, filter_hubs  # ✅ PASSER PARAMÈTRE
        )
        
        ml_scores = [r['ml_score'] for r in recommendations]
        
        return {
            "author_id": author_id,
            "strategy": strategy,
            "filter_hubs_applied": filter_hubs,  # ✅ INDIQUER
            "total_recommendations": len(recommendations),
            "score_stats": {
                "min_ml_score": min(ml_scores) if ml_scores else 0,
                "max_ml_score": max(ml_scores) if ml_scores else 0,
                "avg_ml_score": sum(ml_scores) / len(ml_scores) if ml_scores else 0
            },
            "recommendations": recommendations
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


# ============= AJOUTER ENDPOINT DE TEST =============
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