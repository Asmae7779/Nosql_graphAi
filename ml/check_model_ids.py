import torch
import pickle

def check_model_ids():
    """Vérifier les IDs dans le modèle étendu"""
    model_path = "models/link_prediction_model_extended.pkl"
    
    print("🔍 Vérification des IDs dans le modèle étendu")
    print("=" * 50)
    
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    
    # Afficher les clés
    print(f"📋 Clés disponibles: {list(checkpoint.keys())}")
    
    # Vérifier node_to_idx
    if 'node_to_idx' in checkpoint:
        node_to_idx = checkpoint['node_to_idx']
        print(f"👥 Nombre d'auteurs: {len(node_to_idx)}")
        
        # Afficher les 10 premiers IDs
        sample_ids = list(node_to_idx.keys())[:10]
        print(f"📝 Exemple d'IDs (10 premiers):")
        for i, node_id in enumerate(sample_ids):
            print(f"   {i+1}. {node_id} (type: {type(node_id)})")
        
        # Vérifier si 3308557 est dans la liste
        test_id = "3308557"
        test_id_int = 3308557
        
        print(f"\n🔍 Recherche de l'ID 3308557:")
        print(f"   • Recherche comme string '{test_id}': {test_id in node_to_idx}")
        print(f"   • Recherche comme int {test_id_int}: {test_id_int in node_to_idx}")
        
        # Chercher les IDs qui contiennent "3308557"
        matching_ids = [id for id in node_to_idx.keys() if str(id) == test_id or id == test_id_int]
        print(f"   • IDs correspondants: {matching_ids}")
    
    # Vérifier le graphe
    if 'G_full' in checkpoint:
        G = checkpoint['G_full']
        print(f"\n🕸️  Graphe: {G.number_of_nodes()} nœuds")
        
        # Vérifier si 3308557 est dans le graphe
        if test_id in G or test_id_int in G:
            print(f"✅ ID 3308557 trouvé dans le graphe")
            if test_id in G:
                print(f"   • Comme string: degré = {G.degree(test_id)}")
            if test_id_int in G:
                print(f"   • Comme int: degré = {G.degree(test_id_int)}")
        else:
            print(f"❌ ID 3308557 NON trouvé dans le graphe")
    
    # Vérifier les embeddings
    if 'embeddings' in checkpoint:
        print(f"\n🎯 Embeddings shape: {checkpoint['embeddings'].shape}")

if __name__ == "__main__":
    check_model_ids()