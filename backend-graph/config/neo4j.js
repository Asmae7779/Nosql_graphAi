const neo4j = require("neo4j-driver");

let uri = process.env.NEO4J_URI || "bolt://localhost:7687";
if (uri.startsWith("neo4j://") && !uri.startsWith("neo4j+s://")) {
  uri = uri.replace("neo4j://", "bolt://");
  console.log(`⚠️  URI converti de neo4j:// en bolt:// pour connexion directe`);
}

const database = process.env.NEO4J_DATABASE || "neo4j";
const hasAuth = process.env.NEO4J_USER && process.env.NEO4J_PASSWORD;
const auth = hasAuth
  ? neo4j.auth.basic(process.env.NEO4J_USER, process.env.NEO4J_PASSWORD)
  : neo4j.auth.basic("", "");

const driver = neo4j.driver(uri, auth);

const getSession = () => {
  return driver.session({ database: database });
};

if (!hasAuth) {
  console.log("⚠️  Mode sans authentification - Assurez-vous que Neo4j est configuré avec dbms.security.auth_enabled=false");
}

console.log(`📊 Connexion Neo4j: ${uri} | Base de données: ${database}`);

module.exports = { driver, getSession };
