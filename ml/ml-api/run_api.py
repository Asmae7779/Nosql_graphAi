import uvicorn

if __name__ == "__main__":
    print("🚀 Démarrage API ML sur http://localhost:8000")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)