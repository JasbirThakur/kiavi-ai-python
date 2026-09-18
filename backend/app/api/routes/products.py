import json
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Request, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.db.database import get_db
from app.db import models
from app.db.models import UserRole
from app.core.dependencies import get_current_user, require_knowledge_admin
from app.services.audit import log_audit_event

router = APIRouter(prefix="/api/products", tags=["Product Catalog & SKU 360°"])

class CreateProductRequest(BaseModel):
    sku: str
    name: str
    category: str # Door Hardware, Brakes, Medical Monitors, etc.
    description: Optional[str] = None
    attributes: Optional[Dict[str, Any]] = None # e.g. {"dimensions": "200x50mm", "material": "SS316", "fireRating": "60 min"}
    certifications: Optional[List[str]] = None # e.g. ["EN 1125", "CE Mark"]
    media_urls: Optional[List[str]] = None # e.g. ["/static/cad/DL908.pdf"]
    udi_di: Optional[str] = None # Basic UDI-DI for Medical Devices
    imds_id: Optional[str] = None # IMDS ID for Automotive

class AddCompatibilityRequest(BaseModel):
    compatible_sku: Optional[str] = None
    compatible_model: str # e.g. "Honda Civic 2024-2025" or "Endoscope EM-200"
    compatibility_type: str = "OEM_FITMENT" # OEM_FITMENT, ACCESSORY, REPLACEMENT
    notes: Optional[str] = None

class CompareProductsRequest(BaseModel):
    skus: List[str] # 2 to 4 SKUs to compare

@router.post("", summary="Create a new structured Product SKU")
def create_product(
    req: CreateProductRequest,
    request: Request,
    current_user: models.User = Depends(require_knowledge_admin),
    db: Session = Depends(get_db)
):
    """Creates a new structured product SKU in the organization's catalog."""
    existing = db.query(models.Product).filter(
        models.Product.orgId == current_user.orgId,
        models.Product.sku == req.sku
    ).first()
    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Product with SKU '{req.sku}' already exists")

    product = models.Product(
        orgId=current_user.orgId,
        sku=req.sku,
        name=req.name,
        category=req.category,
        description=req.description,
        attributesJson=json.dumps(req.attributes) if req.attributes else "{}",
        certificationsJson=json.dumps(req.certifications) if req.certifications else "[]",
        mediaUrlsJson=json.dumps(req.media_urls) if req.media_urls else "[]",
        udiDi=req.udi_di,
        imdsId=req.imds_id
    )
    db.add(product)
    db.commit()
    db.refresh(product)

    client_ip = request.client.host if request.client else "unknown"
    log_audit_event(
        db=db,
        org_id=current_user.orgId,
        action="PRODUCT_CREATED",
        resource_type="product",
        resource_id=product.id,
        user=current_user,
        details={"sku": product.sku, "name": product.name, "category": product.category},
        ip_address=client_ip
    )

    return {"status": "success", "message": f"Product '{product.sku}' created", "id": product.id}

@router.get("", summary="List or search products with parametric filtering")
def list_products(
    category: Optional[str] = None,
    query: Optional[str] = None,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Lists products in organization, with optional category and keyword search."""
    q = db.query(models.Product)
    if current_user.role not in [UserRole.PLATFORM_ADMIN, "OWNER", "platform_admin"]:
        q = q.filter(models.Product.orgId == current_user.orgId)

    if category:
        q = q.filter(models.Product.category.ilike(f"%{category}%"))
    if query:
        q = q.filter(
            (models.Product.sku.ilike(f"%{query}%")) |
            (models.Product.name.ilike(f"%{query}%")) |
            (models.Product.description.ilike(f"%{query}%"))
        )

    products = q.order_by(models.Product.sku.asc()).limit(100).all()
    results = []
    for p in products:
        results.append({
            "id": p.id,
            "sku": p.sku,
            "name": p.name,
            "category": p.category,
            "description": p.description,
            "attributes": json.loads(p.attributesJson or "{}"),
            "certifications": json.loads(p.certificationsJson or "[]"),
            "udi_di": p.udiDi,
            "imds_id": p.imdsId,
            "created_at": p.createdAt.isoformat() if p.createdAt else ""
        })
    return {"count": len(results), "products": results}

@router.get("/{sku}", summary="Get SKU 360° profile with full specs and fitment")
def get_product_sku_360(
    sku: str,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Returns complete 360° profile of a product including attributes, CAD links, and compatibilities."""
    q = db.query(models.Product).filter(models.Product.sku == sku)
    if current_user.role not in [UserRole.PLATFORM_ADMIN, "OWNER", "platform_admin"]:
        q = q.filter(models.Product.orgId == current_user.orgId)
    product = q.first()
    if not product:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Product with SKU '{sku}' not found")

    compatibilities = db.query(models.ProductCompatibility).filter(
        models.ProductCompatibility.productId == product.id
    ).all()

    return {
        "id": product.id,
        "sku": product.sku,
        "name": product.name,
        "category": product.category,
        "description": product.description,
        "attributes": json.loads(product.attributesJson or "{}"),
        "certifications": json.loads(product.certificationsJson or "[]"),
        "media_urls": json.loads(product.mediaUrlsJson or "[]"),
        "udi_di": product.udiDi,
        "imds_id": product.imdsId,
        "compatibilities": [
            {
                "id": c.id,
                "compatible_model": c.compatibleModel,
                "compatible_sku": c.compatibleSku,
                "compatibility_type": c.compatibilityType,
                "notes": c.notes
            }
            for c in compatibilities
        ],
        "created_at": product.createdAt.isoformat() if product.createdAt else ""
    }

@router.post("/compare", summary="Side-by-Side Product Specification Comparison Matrix")
def compare_products(
    req: CompareProductsRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Generates a structured side-by-side comparison matrix for 2 to 4 SKUs.
    Extracts common and unique technical attributes across all selected products.
    """
    if len(req.skus) < 2 or len(req.skus) > 4:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Must select between 2 and 4 SKUs for comparison")

    q = db.query(models.Product).filter(models.Product.sku.in_(req.skus))
    if current_user.role not in [UserRole.PLATFORM_ADMIN, "OWNER", "platform_admin"]:
        q = q.filter(models.Product.orgId == current_user.orgId)
    products = q.all()

    if len(products) < len(req.skus):
        found_skus = [p.sku for p in products]
        missing = set(req.skus) - set(found_skus)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Products not found for SKUs: {list(missing)}")

    # Collect all unique attribute keys
    all_attr_keys = set()
    product_data = []
    for p in products:
        attrs = json.loads(p.attributesJson or "{}")
        all_attr_keys.update(attrs.keys())
        product_data.append({
            "sku": p.sku,
            "name": p.name,
            "category": p.category,
            "certifications": json.loads(p.certificationsJson or "[]"),
            "attributes": attrs
        })

    # Build comparison matrix rows
    matrix = []
    # Standard properties
    matrix.append({
        "attribute": "Category",
        "values": {p["sku"]: p["category"] for p in product_data}
    })
    matrix.append({
        "attribute": "Certifications",
        "values": {p["sku"]: ", ".join(p["certifications"]) if p["certifications"] else "None" for p in product_data}
    })
    # Technical parametric attributes
    for key in sorted(all_attr_keys):
        matrix.append({
            "attribute": key,
            "values": {p["sku"]: str(p["attributes"].get(key, "N/A")) for p in product_data}
        })

    return {
        "compared_skus": [p["sku"] for p in product_data],
        "products_summary": [{"sku": p["sku"], "name": p["name"]} for p in product_data],
        "comparison_matrix": matrix
    }

@router.post("/{product_id}/compatibilities", summary="Add compatibility fitment rule to a product")
def add_product_compatibility(
    product_id: str,
    req: AddCompatibilityRequest,
    request: Request,
    current_user: models.User = Depends(require_knowledge_admin),
    db: Session = Depends(get_db)
):
    """Maps a product to a compatible vehicle, machine model, or accessory."""
    product = db.query(models.Product).filter(models.Product.id == product_id).first()
    if not product:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Product not found")

    if current_user.role not in [UserRole.PLATFORM_ADMIN, "OWNER", "platform_admin"] and product.orgId != current_user.orgId:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Unauthorized")

    compat = models.ProductCompatibility(
        orgId=product.orgId,
        productId=product.id,
        compatibleSku=req.compatible_sku,
        compatibleModel=req.compatible_model,
        compatibilityType=req.compatibility_type,
        notes=req.notes
    )
    db.add(compat)
    db.commit()
    db.refresh(compat)

    client_ip = request.client.host if request.client else "unknown"
    log_audit_event(
        db=db,
        org_id=product.orgId,
        action="PRODUCT_COMPATIBILITY_ADDED",
        resource_type="product_compatibility",
        resource_id=compat.id,
        user=current_user,
        details={"product_sku": product.sku, "compatible_model": req.compatible_model},
        ip_address=client_ip
    )

    return {"status": "success", "message": f"Added compatibility for {product.sku} -> {req.compatible_model}", "id": compat.id}

