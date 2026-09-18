"""
seed_european_data.py
Seeds authoritative European Compliance Documents, Chunks, Physical SKUs, and Fitment graphs
for the three primary European enterprise verticals:
1. Healthcare & Medical Devices (EU MDR 2017/745, ISO 13485, UDI-DI)
2. Aerospace & Defense / Heavy Engineering (EASA Part 21/145, EN 9100, Ti-6Al-4V)
3. Automotive & Hardware Manufacturing (IATF 16949, EU CPR 305/2011, EN 1125, IMDS)
"""

import sys
import json
from datetime import datetime, timezone, timedelta

from app.db.database import SessionLocal
from app.db import models
from app.services.embedding import get_embedding

def seed_european_ecosystem():
    db = SessionLocal()
    try:
        # 1. Resolve user organization
        user = db.query(models.User).filter(models.User.email == "jasbirsingh17050@gmail.com").first()
        org_id = user.orgId if user else None
        if not org_id:
            org = db.query(models.Organization).first()
            org_id = org.id if org else None

        print(f"🏢 Seeding European Ecosystem for Org ID: {org_id}")

        # Clean existing seeded European sources and SKUs for idempotency
        existing_sources = db.query(models.BotSource).filter(
            models.BotSource.title.in_([
                "EU MDR Surgical Instruments & Autoclave Sterilization Protocol (Regulation 2017/745)",
                "EASA Part 145 Turbofan Pylon Fasteners & Hydraulic Actuator Maintenance Manual (CMM-71-20)",
                "IATF 16949 High-Performance Braking & EN 1125 Panic Fire Hardware Specification"
            ])
        ).all()
        for es in existing_sources:
            db.query(models.DocumentChunk).filter(models.DocumentChunk.sourceId == es.id).delete()
            db.delete(es)
        db.commit()

        # Clean existing seeded SKUs
        seeded_skus = [
            "MD-CATH-200-EUMDR", "MD-ORTHO-SCR-T5", "MD-STER-TRAY-90",
            "AERO-TF-6AL4V-M10", "AERO-HYD-ACT-450", "AERO-RIV-A286-5",
            "AUTO-CAL-CERAMIC-01", "DL-908-FIRE-EN1125", "AUTO-BOLT-10.9-M12"
        ]
        db.query(models.ProductCompatibility).filter(models.ProductCompatibility.compatibleSku.in_(seeded_skus)).delete(synchronize_session=False)
        existing_products = db.query(models.Product).filter(models.Product.sku.in_(seeded_skus)).all()
        for ep in existing_products:
            db.query(models.ProductCompatibility).filter(models.ProductCompatibility.productId == ep.id).delete(synchronize_session=False)
            db.delete(ep)
        db.commit()

        # =========================================================================
        # 1. SEED EUROPEAN COMPLIANCE DOCUMENTS & KNOWLEDGE CHUNKS
        # =========================================================================
        docs_data = [
            {
                "title": "EU MDR Surgical Instruments & Autoclave Sterilization Protocol (Regulation 2017/745)",
                "category": "Healthcare & Medical Devices",
                "chunks": [
                    (
                        "EU MEDICAL DEVICE REGULATION (EU MDR 2017/745) CLASSIFICATION & REGULATORY SCOPE:\n"
                        "All invasive surgical instruments and reusable surgical devices are governed by EU Regulation 2017/745 Annex VIII. "
                        "Surgical catheters, endoscopic probes, and cannulated instruments are classified under Rule 6 and Rule 7 as Class IIa or Class IIb devices. "
                        "Manufacturers must comply with General Safety and Performance Requirements (GSPR) laid down in Annex I. "
                        "Every medical device requires an assigned Basic UDI-DI (Unique Device Identification) registered within the European database on medical devices (EUDAMED). "
                        "Conformity assessment is audited and verified by EU Notified Body 0123 (TÜV SÜD Product Service GmbH) with active CE marking."
                    ),
                    (
                        "AUTOCLAVE STEAM STERILIZATION PARAMETERS & CYCLE VALIDATION (EN 285 & EN ISO 17665-1):\n"
                        "Sterilization of surgical instruments must strictly satisfy European Standards EN 285 (Large Steam Sterilizers) and EN ISO 17665-1. "
                        "Validated Saturated Steam Sterilization Cycle Parameters:\n"
                        "• Sterilization Temperature: 134°C (273.2°F) with minimum plateau tolerance of +0°C to +3°C.\n"
                        "• Chamber Working Pressure: 3.1 bar (304 kPa) saturated saturated steam.\n"
                        "• Holding / Exposure Time: Exactly 18 minutes minimum for prion decontamination and surgical pathogen elimination.\n"
                        "• Pre-Vacuum Phase: 3 fractional vacuum pulses down to -0.85 bar to eliminate air pockets in lumens.\n"
                        "• Vacuum Drying Phase: 15 minutes at -0.90 bar ensuring zero residual moisture.\n"
                        "• Biological Challenge: Geobacillus stearothermophilus spores (ATCC 7953) with 10^6 spore population demonstrating 100% lethality."
                    ),
                    (
                        "SURGICAL INSTRUMENT CLEANING, PACKAGING & BARRIER INTEGRITY (EN ISO 11607):\n"
                        "Prior to autoclave sterilization, all multi-lumen catheters and instruments must undergo automated ultrasonic cleaning: "
                        "Frequency 40 kHz at 45°C for 15 minutes using enzymatic neutral detergent (pH 7.0 - 8.5), followed by reverse-osmosis water demineralized rinse. "
                        "Packaging must conform to EN ISO 11607-1/2 (Packaging for terminally sterilized medical devices) using medical-grade Tyvek/Mylar or perforated surgical DIN trays (DIN 1.4404 / SS316L). "
                        "Sterile barrier shelf life is verified for 24 months in cleanroom ISO Class 7 storage environments. "
                        "Recommended Official SKU: MD-CATH-200-EUMDR (Steerable Dual-Lumen Catheter, UDI-DI: 08435123456789) and MD-STER-TRAY-90 (SS316L Surgical Autoclave Perforated Tray)."
                    ),
                    (
                        "STERILIZATION WORKFLOW & DECISION FLOWCHART:\n"
                        "The validated multi-step sterilization workflow for surgical devices consists of the following consecutive stages:\n"
                        "1. Post-operative pre-soak within 30 minutes of surgery.\n"
                        "2. Ultrasonic bath cavitation at 40 kHz for 15 minutes.\n"
                        "3. Thermal washer-disinfection at 93°C for 10 minutes (A0 value >= 3000).\n"
                        "4. Optical lumen inspection and borescope integrity verification.\n"
                        "5. EN ISO 11607 Tyvek sterile barrier heat sealing at 180°C.\n"
                        "6. Steam Autoclave Cycle: 134°C @ 3.1 bar for 18 minutes.\n"
                        "7. Post-vacuum pulse dry and chemical integrator verification (Class 5/6 indicator turn black).\n"
                        "8. EUDAMED UDI-DI barcode scan and hospital batch traceability log."
                    )
                ]
            },
            {
                "title": "EASA Part 145 Turbofan Pylon Fasteners & Hydraulic Actuator Maintenance Manual (CMM-71-20)",
                "category": "Aerospace & Defense / Heavy Engineering",
                "chunks": [
                    (
                        "EASA REGULATORY APPROVAL & AIRWORTHINESS RELEASE (EASA PART 21 & PART 145):\n"
                        "Component maintenance, installation, and overhaul of turbofan engine pylon fasteners and flight control actuators must strictly comply with European Aviation Safety Agency (EASA) regulations. "
                        "All manufactured parts must hold EASA Part 21 Subpart G Production Organisation Approval (POA). "
                        "All maintenance actions require an Authorized Release Certificate (EASA Form 1) issued by an approved EASA Part 145 maintenance organization. "
                        "Quality management is certified to European aerospace standard EN 9100 / AS9100 Rev D with 100% lot traceability and EN 10204 Type 3.1 inspection certification."
                    ),
                    (
                        "TITANIUM TURBOFAN PYLON FASTENER SPECIFICATIONS & TORQUE TOLERANCE:\n"
                        "Turbofan structural pylon-to-wing attachment fasteners must withstand high vibration, acoustic shock, and thermal cycling.\n"
                        "• Material Grade: Titanium Alloy Ti-6Al-4V (Grade 5 AMS 4928 / Werkstoff 3.7165).\n"
                        "• Tensile Strength: Minimum 1100 MPa (160 ksi); Shear Strength: Minimum 650 MPa.\n"
                        "• Thread Size & Pitch: M10 x 1.25 metric fine thread, Class 4h6h rolled threads.\n"
                        "• Tightening Torque Specification: Nominal torque is exactly 78.5 Nm with a strict operational tolerance band of ± 2.0 Nm (76.5 Nm to 80.5 Nm).\n"
                        "• Torquing Method: Two-stage torque-angle procedure. First stage: Snug torque to 35 Nm. Second stage: Smooth rotation up to 78.5 Nm ± 2.0 Nm using an ISO 6789 calibrated digital torque wrench.\n"
                        "• Approved Anti-Seize Paste: BMS 3-33 / MIL-PRF-83483 molybdenum disulfide lubricant applied sparingly to threads and washer face.\n"
                        "• Operational Temperature Range: -55°C to +420°C continuous ambient.\n"
                        "• Primary Aircraft Fitment: Airbus A320neo / A321neo CFM LEAP-1A and Pratt & Whitney PW1100G engine pylon mountings.\n"
                        "• Recommended Official SKU: AERO-TF-6AL4V-M10 (EASA Form 1 Certified Pylon Fastener)."
                    ),
                    (
                        "PRIMARY FLIGHT CONTROL HYDRAULIC SERVO-ACTUATOR SPECIFICATIONS:\n"
                        "The primary elevator and aileron flight controls utilize dual-tandem hydraulic servo-actuators operating under severe aerodynamic loads:\n"
                        "• Operating Pressure: Nominal 3000 PSI (207 bar); Proof Pressure: 4500 PSI (310 bar); Burst Pressure: 7500 PSI (517 bar).\n"
                        "• Hydraulic Fluid Standard: Skydrol LD-4 / HyJet IV-A plus fire-resistant phosphate ester hydraulic fluid (BMS 3-11).\n"
                        "• Operating Temperature: -55°C to +135°C fluid temperature.\n"
                        "• Hydraulic Seal Material: Ethylene Propylene Diene Monomer (EPDM / EPR) aerospace grade with back-up PTFE rings.\n"
                        "• Stroke & Force: Stroke length 120 mm; Maximum stall output force 45 kN.\n"
                        "• Overhaul Interval & Reliability: Mean Time Between Failures (MTBF) 25,000 flight hours. TBO (Time Between Overhaul) 12,000 flight cycles.\n"
                        "• Recommended Official SKU: AERO-HYD-ACT-450 (Dual-Tandem Hydraulic Servo-Actuator)."
                    ),
                    (
                        "AEROSPACE FASTENER TORQUE & INSTALLATION WORKFLOW:\n"
                        "The approved EASA Part 145 fastener installation procedure consists of:\n"
                        "1. Visual & Eddy Current non-destructive testing (NDT) of bolt shank and counterbore seat.\n"
                        "2. Degreasing threads with approved solvent (IPA 99.9%).\n"
                        "3. Applying thin coat of BMS 3-33 molybdenum disulfide anti-seize paste.\n"
                        "4. Hand-threading fastener to minimum 3 full turns engagement.\n"
                        "5. Initial snug torque to 35 Nm.\n"
                        "6. Final calibrated tightening to 78.5 Nm ± 2.0 Nm.\n"
                        "7. Installing MS20995C32 stainless steel safety lock-wire or cotter pin.\n"
                        "8. Dual sign-off inspection stamp and EASA Form 1 release generation."
                    )
                ]
            },
            {
                "title": "IATF 16949 High-Performance Braking & EN 1125 Panic Fire Hardware Specification",
                "category": "Automotive & Hardware Manufacturing",
                "chunks": [
                    (
                        "IATF 16949 AUTOMOTIVE BRAKING SYSTEM SPECIFICATIONS & PROCESS QUALITY:\n"
                        "High-performance automotive brake calipers and hydraulic braking components are manufactured under IATF 16949:2016 and VDA 6.3 process quality standards. "
                        "Every safety-critical braking part is registered in the International Material Data System (IMDS ID: IMDS-AUTO-99214) for EU End-of-Life Vehicles (ELV) directive compliance. "
                        "All brake calipers are homologated to European regulation ECE R90.\n"
                        "• Caliper Construction: Monobloc 6-piston caliper CNC-machined from aerospace-grade forged aluminum 6061-T6.\n"
                        "• Piston Configuration: Staggered differential bore titanium pistons (30 mm, 34 mm, 38 mm) to combat uneven brake pad taper wear.\n"
                        "• Caliper Knuckle Mounting Bolt Torque: Exactly 115 Nm using ISO 898-1 Class 10.9 flanged bolts coated in Geomet 500.\n"
                        "• Bleed Screw Torque: 14 Nm (M10x1.0 brass-nickel plated).\n"
                        "• Brake Fluid Specification: High-performance DOT 5.1 glycol-ether fluid. Dry boiling point >= 260°C (500°F); Wet boiling point >= 180°C (356°F).\n"
                        "• Pressure Bleeding Procedure: 1.5 bar (22 PSI) positive pressure bleed sequence: Rear Right -> Rear Left -> Front Right -> Front Left.\n"
                        "• Fitment: Porsche 911 GT3 / Cayman GT4, BMW M3/M4, Audi RS6 Avant with 380mm x 34mm carbon-ceramic brake rotors.\n"
                        "• Recommended Official SKU: AUTO-CAL-CERAMIC-01 (Monobloc 6-Piston Caliper) and AUTO-BOLT-10.9-M12 (Class 10.9 Flanged Chassis Bolt)."
                    ),
                    (
                        "EUROPEAN STANDARD EN 1125 PANIC EXIT DEVICES & EU CPR (305/2011):\n"
                        "Emergency exit hardware for commercial and public buildings across Europe is strictly regulated by EU Construction Products Regulation CPR 305/2011 and Harmonized European Standard EN 1125:2008 (Panic exit devices operated by a horizontal bar).\n"
                        "• Classification Code: EN 1125 3-7-7-B-1-4-2-1-A-A.\n"
                        "• Operational Durability: Grade 7 tested to 200,000 operating cycles without failure or mechanism jamming.\n"
                        "• Egress Panic Release Force: Maximum opening force under unloaded conditions must be less than 80 N (measured at mid-point of horizontal bar). Under a side-load of 1000 N applied to the door leaf, the release force must remain under 220 N.\n"
                        "• Material & Corrosion Resistance: Grade AISI 316 (SS316) marine-grade stainless steel faceplate, internal cams, and deadbolts. EN 1670 Grade 5 very high corrosion resistance (480 hours neutral salt spray test).\n"
                        "• CE Declaration of Performance: CE DoP Number 0432-CPR-0012 certified by MPA NRW Notified Body 0432.\n"
                        "• Recommended Official SKU: DL-908-FIRE-EN1125 (Heavy-Duty Fire Egress Panic Mortise Lock)."
                    ),
                    (
                        "FIRE & SMOKE RESISTANCE CERTIFICATION (EN 1634-1 & DIN 4102-5):\n"
                        "The DL-908-FIRE-EN1125 mortise lock is certified under European fire test standard EN 1634-1:\n"
                        "• Fire Rating Classification: EI 120 and EI 180 (180 minutes fire resistance in full-scale furnace exposure at 1100°C).\n"
                        "• Intumescent Seal Integration: 2mm Graphite-based intumescent pack expands at 180°C to seal lock mortise cavity against smoke migration conforming to EN 1634-3 (Smoke Control Class Sa / Sm).\n"
                        "• Latch Throw: 20 mm single-throw hardened steel deadbolt with integrated anti-saw carbide pins.\n"
                        "• Spindle / Backset Dimensions: 65 mm backset, 72 mm centre distance (PZ Euro-profile cylinder), 9 mm split spindle for fire doors."
                    ),
                    (
                        "HARDWARE & BRAKING VERIFICATION FLOWCHART:\n"
                        "The verification protocol for European Hardware & Automotive parts comprises:\n"
                        "1. Receiving inspection and EN 10204 3.1 material certificate verification.\n"
                        "2. Coordinate Measuring Machine (CMM) dimensional tolerance check (±0.015 mm).\n"
                        "3. IMDS chemical composition registration and RoHS/REACH SVHC audit.\n"
                        "4. Dynamic bench endurance testing: 200,000 cycles under EN 1125 or 500 thermal brake snub cycles.\n"
                        "5. Torque-tension friction coefficient testing (VDA 235-101) verifying torque-to-clamp force.\n"
                        "6. Packaging with CE Marking, DoP documentation, and installation manual in 24 EU languages."
                    )
                ]
            }
        ]

        for doc_item in docs_data:
            source = models.BotSource(
                title=doc_item["title"],
                kind="DOC",
                isUniversal=True,
                orgId=org_id,
                status="APPROVED",
                version="v2.4",
                reviewIntervalDays=365,
                createdAt=datetime.now(timezone.utc)
            )
            db.add(source)
            db.commit()
            db.refresh(source)
            print(f"📄 Created Source: {source.title} (ID: {source.id})")

            for chunk_text in doc_item["chunks"]:
                emb = get_embedding(chunk_text)
                chunk = models.DocumentChunk(
                    sourceId=source.id,
                    content=chunk_text,
                    embedding=emb,
                    document_type="DOC",
                    status="APPROVED",
                    extracted_date=datetime.now(timezone.utc)
                )
                db.add(chunk)
            db.commit()
            print(f"   ↳ Added {len(doc_item['chunks'])} chunks with 384-d vector embeddings.")

        # =========================================================================
        # 2. SEED PHYSICAL EUROPEAN SKUS & SPECIFICATIONS (9 SKUS)
        # =========================================================================
        skus_data = [
            # Healthcare SKUs
            {
                "sku": "MD-CATH-200-EUMDR",
                "name": "Steerable Dual-Lumen Surgical Catheter",
                "category": "Healthcare & Medical Devices",
                "description": "High-precision steerable catheter for endoscopic cardiovascular and vascular access under EU MDR 2017/745 Class IIa.",
                "udiDi": "08435123456789",
                "imdsId": None,
                "certifications": ["CE Mark (CE 0123)", "EU MDR 2017/745", "ISO 13485:2016", "ISO 10993 Biocompatible", "EN ISO 11607"],
                "attributes": {
                    "Material": "Pebax 7233 outer jacket with PTFE low-friction inner liner",
                    "Working Length": "1200 mm",
                    "Outer Diameter": "2.0 mm (6 French)",
                    "Max Burst Pressure": "30 bar (435 PSI)",
                    "Sterilization Standard": "Autoclave 134°C @ 3.1 bar (18 min) / Ethylene Oxide",
                    "Biocompatibility": "ISO 10993 Cytotoxicity & Hemocompatibility Passed",
                    "Notified Body": "TÜV SÜD (CE 0123)",
                    "Risk Class": "Class IIa (EU MDR Annex VIII Rule 6)"
                },
                "compatibilities": [
                    {"model": "Olympus & Karl Storz Endoscopic Sheaths (6 Fr)", "type": "ACCESSORY", "notes": "Seamless fitment through standard 2.2mm working channel"},
                    {"model": "Standard Luer-Lock Surgical Infusion Ports", "type": "OEM_FITMENT", "notes": "Conforms to ISO 80369-7 small-bore connector standards"}
                ]
            },
            {
                "sku": "MD-ORTHO-SCR-T5",
                "name": "Cannulated Titanium Cancellous Bone Screw",
                "category": "Healthcare & Medical Devices",
                "description": "Orthopedic bone fixation screw manufactured from medical-grade Ti-6Al-4V ELI alloy under EU MDR Class IIb.",
                "udiDi": "08435987654321",
                "imdsId": None,
                "certifications": ["CE Mark (CE 0123)", "EU MDR 2017/745", "ISO 13485:2016", "ISO 5832-3", "ASTM F136"],
                "attributes": {
                    "Material": "Titanium Ti-6Al-4V ELI (Grade 23 / ISO 5832-3)",
                    "Length": "45 mm (Available 20mm - 90mm)",
                    "Thread Diameter": "6.5 mm cancellous thread",
                    "Core Cannulation": "2.8 mm guide-wire passage",
                    "Tightening Torque": "4.5 Nm ± 0.3 Nm",
                    "Sterilization Standard": "Steam Autoclave 134°C @ 3.1 bar (EN 285)",
                    "Fatigue Strength": "Runout at 10^7 cycles @ 350 MPa",
                    "Risk Class": "Class IIb (EU MDR Annex VIII Rule 8)"
                },
                "compatibilities": [
                    {"model": "Synthes / Stryker Orthopedic Large Fragment Plating Systems", "type": "OEM_FITMENT", "notes": "Standard 3.5mm hex drive interface"}
                ]
            },
            {
                "sku": "MD-STER-TRAY-90",
                "name": "Heavy-Duty Surgical Autoclave Perforated DIN Tray",
                "category": "Healthcare & Medical Devices",
                "description": "Medical instrument sterilization container tray with silicone retention mats, conforming to DIN 58952 and EN 285.",
                "udiDi": "08435112233445",
                "imdsId": None,
                "certifications": ["CE Mark", "EU MDR 2017/745", "DIN 58952", "EN 285", "ISO 11140-1 Class 5"],
                "attributes": {
                    "Material": "DIN 1.4404 / AISI 316L Stainless Steel Electro-Polished",
                    "Dimensions": "480 mm x 250 mm x 70 mm (1/1 DIN standard)",
                    "Perforation Diameter": "4.0 mm staggered pattern (42% open area)",
                    "Max Temperature": "140°C continuous steam autoclaving",
                    "Autoclave Pressure": "Up to 3.5 bar cyclic steam",
                    "Weight": "1.45 kg",
                    "Corrosion Test": "Passed 500 autoclave cycles without discoloration",
                    "Risk Class": "Class I (EU MDR Annex VIII Rule 1)"
                },
                "compatibilities": [
                    {"model": "Getinge & Belimed Hospital Autoclave Chambers", "type": "OEM_FITMENT", "notes": "Conforms to 1/1 DIN hospital sterilization baskets"}
                ]
            },

            # Aerospace SKUs
            {
                "sku": "AERO-TF-6AL4V-M10",
                "name": "High-Tensile Titanium Turbofan Pylon Fastener",
                "category": "Aerospace & Defense / Heavy Engineering",
                "description": "Structural titanium fastener for aircraft engine pylon-to-wing mounting under EASA Part 21 and EN 9100.",
                "udiDi": None,
                "imdsId": None,
                "certifications": ["EASA Form 1 Authorized Release", "EN 9100:2018", "AS9100D", "AMS 4928", "EN 10204 Type 3.1"],
                "attributes": {
                    "Material": "Titanium Ti-6Al-4V (Grade 5 AMS 4928 / Werkstoff 3.7165)",
                    "Thread Size": "M10 x 1.25 metric fine thread (Class 4h6h rolled)",
                    "Length": "85 mm shank length",
                    "Tensile Strength": "1100 MPa minimum (160 ksi)",
                    "Shear Strength": "650 MPa minimum",
                    "Tightening Torque": "78.5 Nm ± 2.0 Nm (Snug 35 Nm)",
                    "Operating Temperature": "-55°C to +420°C",
                    "Lubricant Spec": "BMS 3-33 / MIL-PRF-83483 MoS2 anti-seize",
                    "Inspection Standard": "100% Fluorescent Penetrant & Eddy Current NDT"
                },
                "compatibilities": [
                    {"model": "Airbus A320neo / A321neo CFM LEAP-1A Engine Pylon Mount", "type": "OEM_FITMENT", "notes": "Direct replacement under Airbus CMM 71-20-04"},
                    {"model": "Airbus A320neo Pratt & Whitney PW1100G Pylon Truss", "type": "OEM_FITMENT", "notes": "Approved standard structural fastener"}
                ]
            },
            {
                "sku": "AERO-HYD-ACT-450",
                "name": "Primary Flight Control Dual-Tandem Hydraulic Servo-Actuator",
                "category": "Aerospace & Defense / Heavy Engineering",
                "description": "Fail-operational dual-tandem electro-hydraulic servo actuator for primary elevator and aileron flight surfaces.",
                "udiDi": None,
                "imdsId": None,
                "certifications": ["EASA Form 1", "EASA CS-25 Airworthiness", "EN 9100", "DO-160G", "DO-178C Level A"],
                "attributes": {
                    "Operating Pressure": "3000 PSI (207 bar) nominal / 4500 PSI proof",
                    "Hydraulic Fluid": "Skydrol LD-4 / HyJet IV-A (BMS 3-11 phosphate ester)",
                    "Seal Material": "Ethylene Propylene Diene Monomer (EPDM) aerospace compound",
                    "Stroke Length": "120 mm",
                    "Stall Force": "45 kN (10,100 lbf)",
                    "Operating Temperature": "-55°C to +135°C fluid / +70°C ambient",
                    "MTBF": "25,000 Flight Hours",
                    "Time Between Overhaul (TBO)": "12,000 Flight Cycles",
                    "Response Bandwidth": "8 Hz @ -3dB"
                },
                "compatibilities": [
                    {"model": "Airbus A330 / A350 Flight Control Hydraulic Surface", "type": "OEM_FITMENT", "notes": "Dual manifold Skydrol Green & Blue hydraulic circuits"}
                ]
            },
            {
                "sku": "AERO-RIV-A286-5",
                "name": "High-Temperature Structural Aircraft Rivet Pin",
                "category": "Aerospace & Defense / Heavy Engineering",
                "description": "Heat-resistant iron-nickel superalloy solid rivet pin for jet exhaust nacelle and turbine heat-shield structures.",
                "udiDi": None,
                "imdsId": None,
                "certifications": ["EASA Part 21", "EN 9100", "AMS 5732", "EN 10204 Type 3.1"],
                "attributes": {
                    "Material": "A-286 Iron-Nickel-Chromium Superalloy (UNS S66286)",
                    "Diameter": "4.8 mm (3/16 inch)",
                    "Tensile Strength": "965 MPa (140 ksi)",
                    "Shear Strength": "620 MPa (90 ksi)",
                    "Operating Temperature": "Up to +700°C continuous",
                    "Corrosion Resistance": "Zero oxidation under turbine exhaust gases",
                    "Installation Tool": "Pneumatic rivet squeezer with 12 kN set force"
                },
                "compatibilities": [
                    {"model": "CFM LEAP-1A / 1B Thrust Reverser & Exhaust Cone", "type": "OEM_FITMENT", "notes": "Approved structural rivet for acoustic hot zone"}
                ]
            },

            # Automotive & Hardware SKUs
            {
                "sku": "AUTO-CAL-CERAMIC-01",
                "name": "Monobloc 6-Piston Carbon-Ceramic Brake Caliper",
                "category": "Automotive & Hardware Manufacturing",
                "description": "High-performance forged aluminum monobloc brake caliper engineered for carbon-ceramic braking under IATF 16949.",
                "udiDi": None,
                "imdsId": "IMDS-AUTO-99214",
                "certifications": ["IATF 16949:2016", "VDA 6.3 Audit Band A", "ECE R90 Certified", "IMDS Registered", "ISO 9001"],
                "attributes": {
                    "Material": "Forged Aerospace Aluminum 6061-T6 CNC-Machined Monobloc",
                    "Piston Count": "6 Staggered Titanium Pistons (30 mm / 34 mm / 38 mm)",
                    "Mounting Torque": "115 Nm (Knuckle mounting bolts M12x1.5 Class 10.9)",
                    "Bleed Screw Torque": "14 Nm (M10x1.0)",
                    "Fluid Compatibility": "DOT 5.1 Glycol-Ether (Dry BP >= 260°C, Wet BP >= 180°C)",
                    "Rotor Compatibility": "380 mm to 410 mm Carbon-Ceramic Matrix (CCM)",
                    "Operating Pressure": "Up to 160 bar hydraulic peak",
                    "Weight": "3.85 kg per caliper (without pads)"
                },
                "compatibilities": [
                    {"model": "Porsche 911 GT3 (992) Front Carbon-Ceramic Rotor 410mm", "type": "OEM_FITMENT", "notes": "Direct radial mount with 225mm bolt spacing"},
                    {"model": "BMW M3 / M4 (G80/G82) M Carbon Ceramic Brake System", "type": "OEM_FITMENT", "notes": "Requires 19 inch or larger track wheels"}
                ]
            },
            {
                "sku": "DL-908-FIRE-EN1125",
                "name": "Heavy-Duty Panic Mortise Lock (EN 1125 & CPR)",
                "category": "Automotive & Hardware Manufacturing",
                "description": "Commercial fire-rated panic escape mortise lock conforming to European Standard EN 1125 and EU CPR 305/2011.",
                "udiDi": None,
                "imdsId": "IMDS-HW-90812",
                "certifications": ["CE Mark (DoP 0432-CPR-0012)", "EN 1125:2008 Grade 7", "EN 12209", "EN 1634-1 (EI 180 min)", "EN 1670 Class 5"],
                "attributes": {
                    "Material": "AISI 316 / SS316 Marine-Grade Stainless Steel",
                    "Backset": "65 mm (Distance from faceplate to keyhole centre)",
                    "Centre Distance": "72 mm (PZ Euro-profile cylinder standard)",
                    "Spindle Size": "9 mm split square spindle",
                    "Panic Opening Force": "< 80 N unloaded / < 220 N under 1000 N door side-load",
                    "Fire Resistance": "EI 120 and EI 180 (180 min at 1100°C furnace test)",
                    "Durability Cycles": "200,000 cycles without failure (EN 1125 Grade 7)",
                    "Corrosion Resistance": "EN 1670 Class 5 (480 hours salt spray tested)",
                    "Deadbolt Throw": "20 mm hardened steel single-throw bolt"
                },
                "compatibilities": [
                    {"model": "DORMA PHA 2500 & GEZE Panic Touchbar Series EN 1125", "type": "OEM_FITMENT", "notes": "Compatible with standard horizontal European touchbars"},
                    {"model": "Commercial Steel & Timber Fire Doors (40mm - 65mm thickness)", "type": "OEM_FITMENT", "notes": "Supplied with intumescent fire insulation pack"}
                ]
            },
            {
                "sku": "AUTO-BOLT-10.9-M12",
                "name": "High-Strength Flanged Chassis Hex Bolt (ISO 898-1)",
                "category": "Automotive & Hardware Manufacturing",
                "description": "Class 10.9 flanged structural chassis bolt with zinc-flake corrosion coating for automotive suspension and hardware mounts.",
                "udiDi": None,
                "imdsId": "IMDS-FAST-4401",
                "certifications": ["IATF 16949", "ISO 898-1 Class 10.9", "VDA 235-104", "RoHS / REACH Compliant"],
                "attributes": {
                    "Material": "Alloy Steel Quenched & Tempered (34Cr4 / 41Cr4)",
                    "Thread Size": "M12 x 1.5 pitch metric thread",
                    "Length": "60 mm under head",
                    "Property Class": "ISO 898-1 Class 10.9 (Tensile 1040 MPa, Yield 940 MPa)",
                    "Surface Coating": "Geomet 500 Zinc-Flake (Passed 720 hours salt spray test)",
                    "Tightening Torque": "125 Nm ± 5 Nm (Friction coefficient µ = 0.12 - 0.15)",
                    "Operating Temperature": "-40°C to +300°C"
                },
                "compatibilities": [
                    {"model": "Automotive Steering Knuckle & Caliper Carrier Bracket", "type": "OEM_FITMENT", "notes": "Direct OEM fitment for high-stress chassis connections"}
                ]
            }
        ]

        for s_item in skus_data:
            product = models.Product(
                orgId=org_id,
                sku=s_item["sku"],
                name=s_item["name"],
                category=s_item["category"],
                description=s_item["description"],
                attributesJson=json.dumps(s_item["attributes"]),
                certificationsJson=json.dumps(s_item["certifications"]),
                mediaUrlsJson=json.dumps([]),
                udiDi=s_item["udiDi"],
                imdsId=s_item["imdsId"],
                createdAt=datetime.now(timezone.utc),
                updatedAt=datetime.now(timezone.utc)
            )
            db.add(product)
            db.commit()
            db.refresh(product)
            print(f"🔧 Seeded SKU: {product.sku} ({product.name}) [{product.category}]")

            # Seed Compatibilities
            for comp in s_item.get("compatibilities", []):
                pc = models.ProductCompatibility(
                    orgId=org_id,
                    productId=product.id,
                    compatibleSku=s_item["sku"],
                    compatibleModel=comp["model"],
                    compatibilityType=comp.get("type", "OEM_FITMENT"),
                    notes=comp.get("notes", ""),
                    createdAt=datetime.now(timezone.utc)
                )
                db.add(pc)
            db.commit()

        # =========================================================================
        # 3. SEED EUROPEAN MULTILINGUAL DOCUMENTS (GERMAN & FRENCH)
        # =========================================================================
        multilingual_docs = [
            {
                "title": "EU MDR Chirurgische Instrumente & Autoklaven-Sterilisationsprotokoll (Verordnung 2017/745)",
                "language": "de",
                "is_ai_translated": False,
                "is_formally_reviewed": True,
                "status": "APPROVED",
                "chunks": [
                    (
                        "EU-MEDIZINPRODUKTE-VERORDNUNG (EU MDR 2017/745) ANWENDUNGSBEREICH:\n"
                        "Invasive chirurgische Instrumente und wiederverwendbare chirurgische Instrumente unterliegen der EU-Verordnung 2017/745 Anhang VIII. "
                        "Chirurgische Katheter, endoskopische Sonden und kanülierte Instrumente sind gemäß Regel 6 und Regel 7 als Medizinprodukte der Klasse IIa oder IIb eingestuft. "
                        "Hersteller müssen die Grundlegenden Sicherheits- und Leistungsanforderungen (GSPR) gemäß Anhang I einhalten. "
                        "Jedes Medizinprodukt benötigt eine zugewiesene Basis-UDI-DI, die in der europäischen Datenbank für Medizinprodukte (EUDAMED) registriert ist."
                    ),
                    (
                        "DAMPFSTERILISATION PARAMETER & ZYKLUSVALIDIERUNG (EN 285 & EN ISO 17665-1):\n"
                        "Die Sterilisation von chirurgischen Instrumenten muss strikt den europäischen Normen EN 285 und EN ISO 17665-1 entsprechen. "
                        "Validierte Heißdampf-Sterilisationsparameter:\n"
                        "• Sterilisationstemperatur: 134°C mit minimaler Haltetoleranz von +0°C bis +3°C.\n"
                        "• Kammerarbeitsdruck: 3,1 bar gesättigter Reindampf.\n"
                        "• Halte- / Einwirkzeit: Genau 18 Minuten zur Prioneninaktivierung und Keimabtötung.\n"
                        "• Vorvakuumphase: 3 fraktionierte Vakuumpulse bis -0,85 bar zur Entlüftung von Hohlkörpern.\n"
                        "• Trocknungsphase: 15 Minuten Vakuumtrocknung bei -0,90 bar."
                    )
                ]
            },
            {
                "title": "EASA Part 145 Turbofan-Pylonbefestigungen Wartungshandbuch (CMM-71-20)",
                "language": "de",
                "is_ai_translated": True,
                "is_formally_reviewed": False,
                "status": "UNDER_REVIEW",
                "chunks": [
                    (
                        "[DEUTSCHE KI-ÜBERSETZUNG - AUSSTEHENDE COMPLIANCE-PRÜFUNG]\n"
                        "EASA BEHÖRDLICHE ZULASSUNG (EASA PART 21 & PART 145):\n"
                        "Wartung, Installation und Überholung von Triebwerkspylon-Verbindungselementen und Flugsteuerungsaktoren müssen den Vorschriften der Europäischen Agentur für Flugsicherheit (EASA) entsprechen. "
                        "Alle gefertigten Komponenten müssen über eine EASA Part 21 Unterabschnitt G Genehmigung als Herstellungsbetrieb (POA) verfügen. "
                        "Jede Freigabe erfordert ein EASA Form 1 Zertifikat."
                    ),
                    (
                        "[DEUTSCHE KI-ÜBERSETZUNG - AUSSTEHENDE COMPLIANCE-PRÜFUNG]\n"
                        "TITAN TURBOFAN PYLON-BEFESTIGUNG DREHMOMENTSPEZIFIKATION:\n"
                        "Strukturelle Pylon-Verbindungselemente aus Titanlegierung Ti-6Al-4V (AMS 4928).\n"
                        "• Gewinde: M10 x 1,25 metrisches Feingewinde.\n"
                        "• Solldrehmoment: Genau 78,5 Nm mit einem Toleranzband von ± 2,0 Nm (76,5 Nm bis 80,5 Nm).\n"
                        "• Zugelassene Montagepaste: BMS 3-33 Molybdändisulfid-Schmierstoff.\n"
                        "• Betriebstemperatur: -55°C bis +420°C ambient."
                    )
                ]
            },
            {
                "title": "Protocole de stérilisation et dispositifs médicaux UE MDR (Règlement 2017/745)",
                "language": "fr",
                "is_ai_translated": True,
                "is_formally_reviewed": False,
                "status": "UNDER_REVIEW",
                "chunks": [
                    (
                        "[TRADUCTION IA EN FRANÇAIS - EN ATTENTE DE VALIDATION CONFORMITÉ]\n"
                        "RÈGLEMENT EUROPÉEN SUR LES DISPOSITIFS MÉDICAUX (UE MDR 2017/745):\n"
                        "Les dispositifs chirurgicaux invasifs et les instruments réutilisables sont régis par l'Annexe VIII du Règlement UE 2017/745. "
                        "Les cathéters et sondes endoscopiques sont classés en Classe IIa ou Classe IIb selon les Règles 6 et 7. "
                        "Chaque dispositif médical requiert un identifiant IUD-ID de base enregistré dans la base de données européenne EUDAMED."
                    )
                ]
            }
        ]

        for m_doc in multilingual_docs:
            source = models.BotSource(
                title=m_doc["title"],
                kind="DOC",
                isUniversal=True,
                orgId=org_id,
                status=m_doc["status"],
                version="v1.0-eu",
                language=m_doc["language"],
                isAiTranslated=m_doc["is_ai_translated"],
                isFormallyReviewed=m_doc["is_formally_reviewed"],
                reviewIntervalDays=365,
                createdAt=datetime.now(timezone.utc)
            )
            db.add(source)
            db.commit()
            db.refresh(source)
            print(f"🌍 Created Multilingual Source [{m_doc['language'].upper()}]: {source.title} (Reviewed: {source.isFormallyReviewed})")

            for chunk_text in m_doc["chunks"]:
                emb = get_embedding(chunk_text)
                chunk = models.DocumentChunk(
                    sourceId=source.id,
                    content=chunk_text,
                    embedding=emb,
                    document_type="DOC",
                    status=m_doc["status"],
                    language=m_doc["language"],
                    isAiTranslated=m_doc["is_ai_translated"],
                    isFormallyReviewed=m_doc["is_formally_reviewed"],
                    extracted_date=datetime.now(timezone.utc)
                )
                db.add(chunk)
            db.commit()

        # =========================================================================
        # 4. SEED CYBER RESILIENCE ACT (CRA) VULNERABILITY & SECURITY REGISTRY
        # =========================================================================
        db.query(models.CRAVulnerability).filter(models.CRAVulnerability.orgId == org_id).delete()
        db.commit()

        cra_vulns = [
            {
                "id": "KIAVI-SEC-001",
                "component": "FastAPI Web Framework & Starlette Exception Handler",
                "severity": "LOW",
                "cvss": 3.1,
                "status": "PATCHED",
                "affected": "< 0.115.0",
                "patched": "0.115.0",
                "advisory": "Upgraded ASGI handler preventing improper header boundary parsing in reverse proxy setups.",
                "sbom": "pkg:pypi/fastapi@0.115.0"
            },
            {
                "id": "KIAVI-SEC-002",
                "component": "Trafilatura DOM Parser & Boilerplate Stripper",
                "severity": "MEDIUM",
                "cvss": 5.3,
                "status": "PATCHED",
                "affected": "< 1.12.0",
                "patched": "1.12.0",
                "advisory": "Hardened regular expression preventing algorithmic complexity backtracking on nested XML comments.",
                "sbom": "pkg:pypi/trafilatura@1.12.0"
            },
            {
                "id": "KIAVI-SEC-003",
                "component": "PostgreSQL pgvector Embedding Distance Indexer",
                "severity": "LOW",
                "cvss": 2.8,
                "status": "MITIGATED",
                "affected": "< 0.3.2",
                "patched": "0.3.2",
                "advisory": "Enforced strict query parameter bounds checking on HNSW vector indexing operators.",
                "sbom": "pkg:pypi/pgvector@0.3.2"
            }
        ]

        for cv in cra_vulns:
            v_rec = models.CRAVulnerability(
                orgId=org_id,
                vulnerabilityId=cv["id"],
                componentName=cv["component"],
                severity=cv["severity"],
                cvssScore=cv["cvss"],
                status=cv["status"],
                affectedVersions=cv["affected"],
                patchedVersion=cv["patched"],
                advisoryText=cv["advisory"],
                sbomComponent=cv["sbom"],
                supportLifecycleUntil=datetime.now(timezone.utc) + timedelta(days=5 * 365)
            )
            db.add(v_rec)
        db.commit()
        print(f"🛡️ Seeded {len(cra_vulns)} Cyber Resilience Act (CRA) Security & Vulnerability Registry records.")

        # =========================================================================
        # 5. SEED EU AI ACT GOVERNANCE & 6 AI CAPABILITIES
        # =========================================================================
        db.query(models.AICapabilityAssessment).filter(models.AICapabilityAssessment.orgId == org_id).delete()
        db.commit()

        ai_capabilities = [
            {
                "name": "AI Search",
                "intended_use": "Strictly grounded factual document search across indexed corporate manuals and websites.",
                "risk_tier": "MINIMAL_RISK",
                "obligations": ["Voluntary Codes of Conduct (Article 95)", "Fact-checking auditability", "Vector integrity"],
                "mdr": "NOT_MEDICAL_DEVICE",
                "mdr_justification": "Search retrieval tool for administrative documents; does not provide patient-specific clinical diagnosis.",
                "oversight": "SME approval gate before documents become searchable."
            },
            {
                "name": "AI Assistant",
                "intended_use": "Conversational Q&A assistant interacting with employees, distributors, and customers.",
                "risk_tier": "LIMITED_RISK",
                "obligations": ["Article 50 Transparency (Notice of AI interaction)", "Machine-generated content disclosure", "Citations traceability"],
                "mdr": "NOT_MEDICAL_DEVICE",
                "mdr_justification": "Provides verified technical specifications and operational guidance.",
                "oversight": "Human-in-the-loop escalation to live representative; Ground Truth locking."
            },
            {
                "name": "AI Research",
                "intended_use": "Multi-document synthesis, cross-standard correlation (EN, ISO, EASA, IATF), and deep technical research.",
                "risk_tier": "LIMITED_RISK",
                "obligations": ["Article 50 Transparency", "Provenance citation locking", "Hallucination escape hatch"],
                "mdr": "NOT_MEDICAL_DEVICE",
                "mdr_justification": "Technical research engine; explicit disclaimer that engineering sign-off is mandatory.",
                "oversight": "Exported research dossiers require formal engineer sign-off."
            },
            {
                "name": "AI Learning Assistant",
                "intended_use": "Corporate LMS Academy interactive tutor, lesson summarization, and automated quiz evaluation.",
                "risk_tier": "LIMITED_RISK",
                "obligations": ["Bias prevention in scoring", "Transparency of automated grading criteria", "Human instructor grade override"],
                "mdr": "NOT_MEDICAL_DEVICE",
                "mdr_justification": "Corporate training and staff upskilling.",
                "oversight": "Human corporate trainer review for certificate issuance."
            },
            {
                "name": "AI Agents",
                "intended_use": "Autonomous multi-step workflows, catalog verification, lead generation, and CRM synchronization.",
                "risk_tier": "LIMITED_RISK",
                "obligations": ["Human-in-the-loop confirmation for high-impact actions", "Activity logging in immutable audit trail", "Kill-switch mechanism"],
                "mdr": "NOT_MEDICAL_DEVICE",
                "mdr_justification": "Industrial process orchestration.",
                "oversight": "Deterministic action permission boundaries with emergency stop."
            },
            {
                "name": "AI-Supported Automation",
                "intended_use": "Automated technical spec extraction, SKU compatibility graph generation, and regulatory compliance mapping.",
                "risk_tier": "LIMITED_RISK",
                "obligations": ["Technical documentation maintenance", "Data governance & validation of training sets", "Continuous post-market monitoring"],
                "mdr": "NOT_MEDICAL_DEVICE",
                "mdr_justification": "Provides operational decision-support for engineers; not intended for direct clinical patient monitoring.",
                "oversight": "SME approval workflow for all generated SKU graphs and catalogues."
            }
        ]

        for cap in ai_capabilities:
            assessment = models.AICapabilityAssessment(
                orgId=org_id,
                capabilityName=cap["name"],
                intendedUse=cap["intended_use"],
                riskTier=cap["risk_tier"],
                euAiActObligationsJson=json.dumps(cap["obligations"]),
                mdrClassification=cap["mdr"],
                mdrRule11Justification=cap["mdr_justification"],
                isCustomerFacing=True,
                humanOversightMeasures=cap["oversight"],
                watermarkingEnabled=True,
                assessedBy=user.email if user else "compliance@kiavi-ai.eu"
            )
            db.add(assessment)
        db.commit()
        print(f"⚖️ Seeded EU AI Act Governance & MDR assessments across all 6 core AI capabilities.")

        print("\n✨ [SUCCESS]: Complete European Regulatory Ecosystem successfully seeded into PostgreSQL database!")
        print(f"   • Core Documents & Multilingual Collections (English, German, French)")
        print(f"   • Cyber Resilience Act (CRA) Security Registry & 5-Year Lifecycle Support")
        print(f"   • EU AI Act Governance & MDR Rule 11 Classification across 6 AI Capabilities")
        print(f"   • 9 Physical European SKUs across Healthcare, Aerospace, and Automotive")

    except Exception as e:
        db.rollback()
        print(f"❌ Error seeding European data: {e}")
        raise e
    finally:
        db.close()

if __name__ == "__main__":
    seed_european_ecosystem()

