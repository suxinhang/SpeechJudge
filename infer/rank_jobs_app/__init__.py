"""
Rank-jobs service package.

Directory layout:
- ``app/``        FastAPI application factory + lifespan
- ``api/``        Routers
- ``core/``       Settings / shared constants
- ``db/``         JSON file job store (no database server)
- ``services/``   Workers + model runtime helpers
"""
