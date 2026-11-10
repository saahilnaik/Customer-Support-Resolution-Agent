"""API-key authentication and role-based authorization.

Clients send `X-API-Key: <key>`. Keys are random tokens; only their SHA-256 is stored
(a fast hash is fine for high-entropy secrets, unlike passwords).

Roles:
- customer: bound to one customer_id; may chat as that customer and see own tickets.
- reviewer: may list/decide approvals and see all tickets; may not chat.

Create a key (printed once, store it safely):
    python -m app.auth create-key --role reviewer --name "Dana"
    python -m app.auth create-key --role customer --name "alice-app" --customer C001
"""
import argparse
import hashlib
import secrets
from dataclasses import dataclass

from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader

from app.db.models import ApiKey, Customer
from app.db.session import session_scope

ROLES = {"customer", "reviewer"}

DEMO_KEYS = [  # (key, role, name, customer_id) - seeded only if SEED_DEMO_KEYS=true
    ("demo-key-alice", "customer", "Alice (demo)", "C001"),
    ("demo-key-bob", "customer", "Bob (demo)", "C002"),
    ("demo-key-carla", "customer", "Carla (demo)", "C003"),
    ("demo-key-reviewer", "reviewer", "Demo Reviewer", None),
]

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False,
                               description="Customer or reviewer API key")


@dataclass(frozen=True)
class Principal:
    role: str
    name: str
    customer_id: str | None = None


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def _validate(role: str, customer_id: str | None) -> None:
    if role not in ROLES:
        raise ValueError(f"role must be one of {sorted(ROLES)}")
    if role == "customer" and not customer_id:
        raise ValueError("customer keys need a customer_id")
    if role == "reviewer" and customer_id:
        raise ValueError("reviewer keys must not have a customer_id")


def _store(s, key: str, role: str, name: str, customer_id: str | None) -> None:
    if customer_id and not s.get(Customer, customer_id):
        raise ValueError(f"customer {customer_id} not found")
    s.add(ApiKey(key_hash=hash_key(key), role=role, name=name, customer_id=customer_id))


def create_api_key(role: str, name: str, customer_id: str | None = None) -> str:
    """Create and store a new key; returns the plaintext key (shown only once)."""
    _validate(role, customer_id)
    key = f"sk_{role[:4]}_{secrets.token_urlsafe(32)}"
    with session_scope() as s:
        _store(s, key, role, name, customer_id)
    return key


def seed_demo_keys() -> None:
    """Idempotent. Demo keys are public knowledge - never enable this in a real deployment."""
    with session_scope() as s:
        for key, role, name, cid in DEMO_KEYS:
            if not s.query(ApiKey).filter(ApiKey.key_hash == hash_key(key)).first():
                _store(s, key, role, name, cid)


# ---- FastAPI dependencies ----

def get_principal(key: str | None = Security(_api_key_header)) -> Principal:
    if not key:
        raise HTTPException(401, "missing API key (X-API-Key header)")
    with session_scope() as s:
        row = s.query(ApiKey).filter(ApiKey.key_hash == hash_key(key)).first()
        if not row or not row.active:
            raise HTTPException(401, "invalid or revoked API key")
        return Principal(role=row.role, name=row.name, customer_id=row.customer_id)


def require_customer(p: Principal = Depends(get_principal)) -> Principal:
    if p.role != "customer":
        raise HTTPException(403, "this endpoint requires a customer key")
    return p


def require_reviewer(p: Principal = Depends(get_principal)) -> Principal:
    if p.role != "reviewer":
        raise HTTPException(403, "this endpoint requires a reviewer key")
    return p


# ---- CLI ----

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m app.auth")
    sub = parser.add_subparsers(dest="cmd", required=True)
    ck = sub.add_parser("create-key", help="create an API key")
    ck.add_argument("--role", required=True, choices=sorted(ROLES))
    ck.add_argument("--name", required=True)
    ck.add_argument("--customer", dest="customer_id")
    rk = sub.add_parser("revoke", help="deactivate every key with this name")
    rk.add_argument("--name", required=True)
    args = parser.parse_args(argv)

    from app.db.session import init_db

    init_db()
    if args.cmd == "create-key":
        try:
            key = create_api_key(args.role, args.name, args.customer_id)
        except ValueError as e:
            parser.error(str(e))
        print(key)
        print("Store this key now; it cannot be shown again.")
    else:
        with session_scope() as s:
            rows = s.query(ApiKey).filter(ApiKey.name == args.name, ApiKey.active).all()
            for r in rows:
                r.active = False
        print(f"revoked {len(rows)} key(s) named {args.name!r}")


if __name__ == "__main__":
    main()
