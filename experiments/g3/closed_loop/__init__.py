"""§4.2 Closed-loop serving package.

Components:
  - flowcache_connector.py: FlowCacheConnector (KVConnectorBase_V1) with selective migration
  - serving_harness.py: vLLM serving harness + trace replayer + metrics collector
  - run_closed_loop.py: CLI runner for 4-strategy closed-loop comparison
"""
