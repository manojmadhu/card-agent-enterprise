# Optional Git-managed cards

Cards are normally managed through the API (`PUT /v1/cards/{id}` + reference uploads, see docs/API.md).
For programmes that need review in Git, put them here and import them with
`python tools/sync_registry.py` or `POST /v1/registry/sync` (incremental; API-owned cards are never touched):

```
registry/cards/<cc>/<programme>.yaml          card definition (schema: src/cardagent/registry/models.py)
registry/refs/<family_id>/<variant_id>/*.png  reference images (png/jpg/webp), 3-20 per variant
```
Templates: `examples/cards/`. Empty on purpose: the platform starts without any card.
