from datetime import datetime, timedelta, timezone

from app.config import get_settings
from app.db.models import Customer, Order
from app.db.session import session_scope


def seed() -> None:
    """Idempotent demo data (customers/orders, plus demo API keys if enabled)."""
    _seed_customers()
    if get_settings().seed_demo_keys:
        from app.auth import seed_demo_keys  # avoid import cycle at module load

        seed_demo_keys()


def _seed_customers() -> None:
    now = datetime.now(timezone.utc)
    with session_scope() as s:
        if s.get(Customer, "C001"):
            return
        s.add_all(
            [
                Customer(id="C001", name="Alice Johnson", email="alice@example.com", tier="gold"),
                Customer(id="C002", name="Bob Smith", email="bob@example.com", tier="standard"),
                Customer(id="C003", name="Carla Gomez", email="carla@example.com", tier="standard"),
            ]
        )
        s.add_all(
            [
                Order(id="ORD-1001", customer_id="C001", items="Wireless Headphones", total=129.99,
                      status="delivered", placed_at=now - timedelta(days=10)),
                Order(id="ORD-1002", customer_id="C001", items="USB-C Cable (2-pack)", total=19.99,
                      status="shipped", placed_at=now - timedelta(days=3)),
                Order(id="ORD-2001", customer_id="C002", items="Mechanical Keyboard", total=89.00,
                      status="delivered", placed_at=now - timedelta(days=45)),
                Order(id="ORD-2002", customer_id="C002", items="Phone Case", total=24.50,
                      status="delivered", placed_at=now - timedelta(days=5)),
                Order(id="ORD-3001", customer_id="C003", items="Smart Watch", total=249.00,
                      status="processing", placed_at=now - timedelta(days=1)),
            ]
        )
