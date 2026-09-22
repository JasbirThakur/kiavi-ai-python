import uuid
import datetime
from app.db.database import SessionLocal
from app.db import models
from app.services.chunking import chunk_text
from app.services.embedding import get_embedding

ORG_ID = "0c3edf38-e7e2-457f-b968-cbe02fe34ff6"

DOC_1_TITLE = "Medical Equipment Engineering and Manufacturing Standard and Policies (Chapter 10).pdf"
DOC_1_CONTENT = """Chapter 10: Medical equipment engineering and manufacturing standard and policies
Authors: Soon Cheng Yap and Wai Yie Leong (Graduate School of Medicine, Perdana University, Malaysia)

The manufacture of medical devices has included the design of the hardware and software which is required to comply with the standards that have been set by the international authorities such as FDA or IEC standards. This is to ensure that the final product that is manufactured by the medical equipment manufacturer makes a safe and effective use of the devices to the end user. Throughout the manufacturing process, the risk assessment is one of the important keys to the success of launching medical products. Assessments and testing during manufacturing verify device performance to fulfill safety requirements such as electrical safety test, electromagnetic compatibility (EMC) or GUI testing.

10.1 Introduction on medical equipment engineering
Engineering on medical equipment applies engineering principles infused with medicine to develop innovative medical devices through new and advanced technology. The manufacture of medical equipment serves diagnostic, therapeutics, and monitoring purposes, with workflow innovations such as Internet of Things (IoT), cloud computing, and big data integrated into the health information system (HIS).

10.2 Manufacturing failure on medical devices
Medical graded wearable monitoring devices monitor heart-related bioactivity such as arrhythmia with ECG and BP analysis using artificial intelligence (AI). Failures on medical devices lead to lawsuits and permanent patient injury. Good manufacturing practice (GMP) is the quality guidance before market launch. ISO 13485 (EU) and 21 CFR Part 820 (FDA) are harmonized standards.
Table 10.1: 15 subsections under FDA 21 CFR Part 820:
- General provisions
- Quality systems requirements
- Design controls
- Documents controls
- Purchasing controls
- Identification and traceability
- Production and process controls
- Acceptance activities
- Nonconforming product
- Corrective and preventive actions (CAPA)
- Labeling and packaging control
- Handling, storage, distribution, and installation
- Records
- Servicing
- Statistical techniques

Case Study of Medical Negligence (2011): An automated non-invasive blood pressure (NIBP) device cuff malfunctioned and remained inflated for more than 5 hours during surgery. The mean arterial pressure (MAP) was supposed to show 60 mmHg in inflation indicating venipuncture mode for intravenous catheterization. A false communication error between software and hardware display occurred without triggering an alarm or error message, causing venous stasis and resulting in the patient's arm being amputated.
Figure 10.1 shows the M3 vital signs monitor by Edan USA integrating NIBP, SpO2, and pulse rate (PR).
Figure 10.2 shows hospital adverse events incidence: Australia (16.6%), New Zealand, Sweden, USA (Boston), UK (London), Spain, Canada, Netherlands, USA (New York, Utah & Colorado), ranging between 4% and 17%.
Figure 10.3 shows medical device recalls from 2005 to 2021: peaking at 1,277 cases in 2011 and dropping to 304 in 2012.
In Japan, Planned Preventative Maintenance (PPM) is managed by the Medical Equipment Safety Manager (MESM) for ventilators, artificial heart/lung apparatus, biventricular assist devices, hemodialysis, infant incubators, and defibrillators.

10.3 Methodologies on existing analysis on medical equipment management
Failure Modes and Effect Analysis (FMEA) identifies potential failure modes, evaluates, and mitigates risks. It indicates:
- Severity (S): effect of seriousness (1 to 10 scale).
- Occurrence (O): probability of frequency (1 to 10 scale).
- Detectability (D): ability to detect cause of failure mode (1 to 10 scale).
Risk Priority Number (RPN) = Severity (S) * Occurrence (O) * Detectability (D).
Table 10.2 Indication of Severity, Occurrence, Detectability (Scores 1-10):
- Score 1: None / Extremely remote / Almost certain
- Score 2: Very minor / Remote, very unlikely / Very high
- Score 3: Minor / Very slight chance / High
- Score 4: Very low / Slight chance / Moderately high
- Score 5: Low / Occasion / Moderate
- Score 6: Moderate / Moderate / Low
- Score 7: High / Frequent / Very low
- Score 8: Very high / High / Remote
- Score 9: Hazardous with warning / Very high / Very remote
- Score 10: Hazardous without warning / Extremely high / Absolutely uncertainty
Modified FMEA in healthcare is known as HFMEA.

10.4 Impact on medical equipment management
Human factor engineering (HFE) applies ergonomics to medical device design interfaces. Standards: IEC 62366 and HE75:2009 minimize human error (inattention, training gaps). Root cause analysis (RCA) investigates incident causal chains.
Figure 10.4 shows hospital adverse events: peak of 250,000 cases reported to FDA in 2011.

10.5 Introduction on manufacturing standard and policies
British Hypertension Society cuff size recommendations:
- Adult arms: 12 x 26 cm
- Obese arms: 12 x 40 cm
- Lean adult and children arms: 12 x 18 cm
Standard IEC 80601-2-30 covers electrical safety requirements and performance for automated non-invasive sphygmomanometers.
IEC standards: EMC emission (IEC 61000), Human factors (IEC 62366), Electrical safety (IEC 60601). WTO eliminates trade barriers for IEC recognized standards.

10.6 Standards on Quality Management System (QMS)
ISO 9001:2015 uses Plan-Do-Check-Act (PDCA) cycle.
Table 10.3 ISO 9001:2015 clauses: Clause 0-3 (Intro & Scope), Clause 4 (Context), Clause 5 (Leadership), Clause 6 (Planning), Clause 7 (Support), Clause 8 (Operation), Clause 9 (Performance Evaluation), Clause 10 (Improvement).

10.6.1 ISO 13485: Medical devices QMS framework. Key elements include Supplier Corrective Action Request (SCAR) and Corrective and Preventive Actions (CAPA) to resolve recurring quality defects, software requirements, packaging, distribution, and post-market surveillance.
10.6.2 ISO 14971: Risk management framework for medical device lifecycle. Workflow: Risk management plan -> Risk analysis + evaluation -> Risk assessment (RPN) -> Risk control -> Risk acceptability -> Review & Report -> Production information.

10.7 Technical standard for safety measurement
10.7.1 Electromagnetic Compatibility (EMC)
Table 10.4 EMC Standards:
- Harmonic current limits: IEC/EN 61000-3-2 (mains 220V, 16A phase current)
- Voltage fluctuation & flicker limitation: IEC/EN 61000-3-3 (low voltage <= 16A; Pst <= 1.0, Plt <= 0.65, dmax <= 4%, 6%, 7%)
- Electrostatic discharge (ESD): IEC/EN 61000-4-2 (Level 4 most severe ESD threat: 8 kV contact discharge, 15 kV air discharge)
- Radiated RFI immunity: IEC/EN 61000-4-3
- Electrical fast transients: IEC/EN 61000-4-4
- Mains surges: IEC/EN 61000-4-5
- Conducted RFI: IEC/EN 61000-4-6
- Mains frequency magnetic field: IEC/EN 61000-4-8
- Pulsed magnetic field: IEC/EN 61000-4-9
- Damped oscillatory magnetic field: IEC/EN 61000-4-10
- Voltage dips and interruptions: IEC/EN 61000-4-11

10.7.2 Electrical Safety: IEC 60601-1 and IEC 60601-2
Defines Means of Protection (MOP):
- MOOP: Means of Operator Protection
- MOPP: Means of Patient Protection
Device Classifications:
- Type B (body): Vicinity within 6-foot radius without patient contact (e.g. X-ray, MRI imaging). Isolation: 1,500 Vac, Creepage: 2.5 mm, Insulation: Basic.
- Type BF (body floating): Physical contact with patient (e.g. blood pressure monitoring). Isolation: 3,000 Vac, Creepage: 5.0 mm, Insulation: Double.
- Type CF (cardiac floating): Direct physical contact with the heart (e.g. Defibrillator). Isolation: 4,000 Vac, Creepage: 8.0 mm, Insulation: Double.
IEC 60601 3rd edition requirements:
- One MOOP: 1,500 Vac isolation, 2.5 mm creepage, Basic insulation.
- Two MOOP: 3,000 Vac isolation, 5.0 mm creepage, Double insulation.
- One MOPP: 1,500 Vac isolation, 4.0 mm creepage, Basic insulation.
- Two MOPP: 4,000 Vac isolation, 8.0 mm creepage, Double insulation.
Figure 10.7 Defibrillator markings: Class I, Class II, Earth reference point, CE Conformité Européenne marking, Type B applied part, Type BF applied part, Type CF applied part, and Defibrillation-proof Type B, BF, and CF parts.

10.8 Standard of Human Factors Engineering (HFE)
IEC 62366 / AAMI TIR59:2017: Harmonized usability engineering standard between EU and US FDA. Figure 10.8 timeline spans AAMI-HE48 (1993), ANSI/AAMI HE74 (2001), IEC 60601-1-6 (2005/2010), IEC 62366:2007, ANSI/AAMI HE75 (2009/R2018), IEC 62366-1:2015, and IEC/TR 62366-2:2016.

10.9 Medical device software: ANSI/AAMI/IEC 62304
Medical device software life cycle processes classified into Class A, Class B, and Class C based on severity of risk. Covers Software as Medical Device (SaMD) like EEG acquisition and mobile health apps.
Figure 10.9 shows computer-related medical device recalls peaked at 319 cases in 2011.
Table 10.7: 11 Software Life Cycle Processes under IEC 62304:
1. Software risk management process
2. Software development planning
3. Software requirement analysis
4. Software architectural design
5. Software detailed design
6. Software unit implementation
7. Software integration and integration testing
8. Software system testing
9. Software release
10. Software configuration management
11. Software problem resolution
"""

DOC_2_TITLE = "CarParts.com Automotive Replacement Parts Catalogue and Directory.pdf"
DOC_2_CONTENT = """CarParts.com Automotive Replacement Parts Catalogue and Directory
Official Parts Directory & Corporate Information

Directory of Automotive Parts:
- 0-9: 4WD Actuator, 4WD Hub Locking Solenoid.
- A: A Arm Bumper, A/C & Heater Control, A/C Accumulator, A/C Actuator, A/C Bracket, A/C Clutch, A/C Compressor, A/C Compressor By-Pass Pulley, A/C Compressor Clutch, A/C Condenser, A/C Condenser Fan, A/C Control Unit, A/C Evaporator, A/C Expansion Valve, A/C Hose, A/C Idler Pulley, A/C Receiver Drier, A/C Refrigerant Hose, A/C Speed Sensor, A/C Switch, A/C Thermo Switch, ABS Cable Harness, ABS Control Module, ABS Hydraulic Unit, ABS Speed Sensor, Accelerator Pedal Position Sensor, Accessory Belt Idler Pulley, Accessory Belt Tension Pulley, Accessory Belt Tensioner, Accessory Belt Tensioner Kit, Accessory Drive Belt, Adjustable Clutch Rod, Air Bag Clockspring, Air Bag Sensor, Air Box, Air Box Thermostat, Air Deflector, Air Filter, Air Filter Heat Shield, Air Inject Check Valve, Air Intake Duct, Air Intake Hose, Air Pump, Air Pump Control Valve, Air Spring, Air Suspension Compressor, Air Suspension Control Valve, Air Temperature Sensor, Alternator, Alternator Brush Set, Alternator Pulley, Ambient Temperature Sensor, Antenna, Antenna Extension Cable, Antenna Mast, Antenna Mount Bushing, Arm Rest, Automatic Transmission Conductor Plate, Automatic Transmission Dipstick, Automatic Transmission Filter, Automatic Transmission Oil Pressure Switch, Automatic Transmission Shift Kit, Automatic Transmission Solenoid, Auxiliary Fan, Auxiliary Water Pump, Axle Assembly, Axle Shaft, Axle Shaft Bearing, Axle Support Bushing.
- B: Back Up Light, Back Up Light Switch, Balance Shaft Kit, Ball Joint, Battery Cable, Battery Hold Down, Battery Tray, Bed Liner, Bed Mat, Bed Rail Cap, Bed Rails, Bike Rack, Billet Grille, Blower Control Switch, Blower Motor, Blower Motor Resistor, Body Armor, Body Control Module, Body Kit, Body Lift Gap Guard, Body Lift Kit, Body Mount Kit, Body Panel, Body Wiring Harness, Boost Pressure Valve, Brake Adjusting Plug, Brake Booster, Brake Caliper, Brake Caliper Bracket, Brake Caliper Repair Kit, Brake Disc, Brake Disc and Pad Kit, Brake Drum, Brake Dust Shields, Brake Hardware Kit, Brake Hose, Brake Light Switch, Brake Line, Brake Master Cylinder, Brake Pad Sensor, Brake Pad Set, Brake Reservoir, Brake Shoe Set, Breather Hose, Brush Guard, Bug Shield, Bulb Socket, Bull Bar, Bump Stop, Bumper, Bumper Absorber, Bumper Bracket, Bumper Cover, Bumper End, Bumper Filler, Bumper Grille, Bumper Guard, Bumper Guide, Bumper Protector, Bumper Reflector, Bumper Reinforcement, Bumper Retainer, Bumper Step Pad, Bumper Trim.
- C: CV Boot, CV Joint, Cab Corner, Cab Cover, Cabin Air Filter, Cam Gear, Cam Phaser, Camber and Alignment Kit, Camshaft Position Sensor, Camshaft Seal, Camshaft Synchronizer, Car Bra, Car Cover, Carburetor, Carburetor Repair Kit, Cargo Bar, Cargo Mat, Carpet, Carpet Kit, Catalytic Converter, Center Bearing, Center Link, Climate Control Unit, Clips & Fasteners, Clutch Disc, Clutch Interlock Switch, Clutch Kit, Clutch Master Cylinder, Clutch Rod, Clutch Slave Cylinder, Coil Over Kit, Coil Spring Conversion Kit, Coil Spring Insulator, Coil Springs, Cold Air Intake, Column Clock Spring, Combination Switch, Connectors, Console, Console Latch, Console Lid, Control Arm, Control Arm Bushing, Control Arm Kit, Control Arm Shaft Kit, Coolant Air Bleeder Kit, Coolant Bypass Line, Coolant Level Sensor, Coolant Reservoir, Coolant Reservoir Cap, Coolant Temperature Sensor, Cooling Fan Assembly, Cooling Fan Hub, Cooling Hose Connector, Cooling Hose Flange, Corner Light, Cowl Hood, Crankcase Vent Hose, Crankcase Vent Valve, Crankshaft Position Sensor, Crankshaft Pulley, Crankshaft Seal Cover Gasket, Crossmember, Cruise Control Switch, Cup Holder, Cylinder Head, Cylinder Head Bolt, Cylinder Head Gasket.
- D: Dash Cover, Dash Knob Kit, Dash Lamp Kit, Dash Trim, Differential Cover, Differential Rebuild Kit, Dimmer Switch, Distributor, Distributor Cap, Distributor Rotor, Door Check, Door Glass Weatherstrip, Door Guard, Door Handle, Door Handle Cover, Door Handle Latch, Door Handle Trim, Door Hinge, Door Hinge Repair Kit, Door Jamb Switch, Door Latch Cable, Door Lock, Door Lock Actuator, Door Lock Cylinder, Door Lock Switch, Door Molding and Beltlines, Door Panel, Door Pull Strap, Door Seal, Door Shell, Door Skin, Door Striker Pin, Door Weatherstrip Seal, Down Pipe, Drag Link, Drive Belt, Drive Shaft Flex Joint, Driveshaft, Driveshaft CV Joint, Driveshaft Pinion Yoke, Driving Light, Driving Light Bracket.
- E: EGR Cooler, EGR Cooler Gasket, EGR Line, EGR Line Fitting, EGR Pressure Feedback Sensor, EGR Vacuum Controller, EGR Vacuum Solenoid, EGR Valve, EGR Valve Gasket, EGR Valve Position Sensor, Emblem, Engine Control Module, Engine Dress Up Kit, Engine Gasket Set, Engine Hardware Kit, Engine Long Block, Engine Shock Mount, Engine Splash Shield, Engine Torque Mount, Exhaust Clamp, Exhaust Gasket, Exhaust Hanger, Exhaust Manifold, Exhaust Manifold Gasket, Exhaust Manifold Gasket Set, Exhaust Pipe, Exhaust Pipe Gasket, Exhaust System, Exhaust Tip, Exhaust Valve, Exterior Door Handle.
- F: Fan Blade, Fan Clutch, Fan Motor, Fan Pulley Bracket, Fan Shroud, Fan Switch, Fender, Fender Flares, Fender Liner, Fender Molding, Fender Support, Fender Trim, Fender Vents, Flex Plate, Floor Mats, Floor Pan, Flywheel, Fog Light, Fog Light Bracket, Fog Light Bulb, Fog Light Cover, Fog Light Trim, Fuel Door Bumper, Fuel Filler Hose, Fuel Filler Neck, Fuel Filler Neck Protector, Fuel Filter, Fuel Injection Wiring Harness, Fuel Injector, Fuel Injector Seal, Fuel Level Sensor, Fuel Line, Fuel Pressure Regulator, Fuel Pressure Sensor, Fuel Pump, Fuel Pump Driver Module, Fuel Pump Relay, Fuel Pump Strainer, Fuel Rail, Fuel Sending Unit, Fuel Tank, Fuel Tank Filler Neck, Fuel Tank Selector Switch, Fuel Tank Strap, Fuel Tank Vent Valve.
- G: GPS Tracking Device, Gas Cap, Glow Plug, Gooseneck Hitch, Grab Handle, Grille Assembly, Grille Bracket, Grille Guard, Grille Insert, Grille Molding, Grille Reinforcement, Grille Shell, Grille Trim.
- H: HID Bulb Ballast, HVAC Heater Blend Door Actuator, Harmonic Balancer, Hazard Flasher Switch, Head Gasket Set, Headache Rack, Header Panel, Header Pipe, Headers, Headlight, Headlight Bezel, Headlight Bracket, Headlight Bulb, Headlight Conversion Kit, Headlight Cover, Headlight Door, Headlight Filler, Headlight Housing, Headlight Lens, Headlight Level Sensor, Headlight Molding, Headlight Motor, Headlight Retainer, Headlight Switch, Headlight Washer Cover, Headliner, Heater Blend Door Actuator, Heater Bypass Valve, Heater Core, Heater Hose, Heater Hose Fitting, Heater Pipe Line, Heater Valve, Helper Spring, Hitch, Hitch Wiring Kits, Hood, Hood Cable, Hood Catch, Hood Hinge, Hood Latch, Hood Molding, Hood Scoop, Horn, Horn Button, Hourmeter, Hub Cap, Hydraulic Timing Belt Actuator.
- I: IAT Sensor, Idle Control Motor, Idle Control Valve, Idler Arm, Idler Arm Bracket, Ignition Coil, Ignition Coil Wire, Ignition Lock Assembly, Ignition Lock Cylinder, Ignition Lock Housing, Ignition Module, Ignition Switch, Instrument Panel Cover, Intake Manifold, Intake Manifold Gasket, Intake Manifold Runner Valve, Intake Plenum Gasket, Intake Tube, Intercooler, Intercooler Hose, Interior Door Handle, Interior Restoration Kit, Interior Trim Kit, Intermediate Shaft, Inverter Cooler.
- J: Jack Pad, Jack Plug Cover.
- K: King Pin Repair Kit, Knock Sensor, Knock Sensor Harness.
- L: LED Light Bar, Lateral Link, Leaf Spring, Leaf Spring Bushing, Leaf Spring Hanger, Leaf Spring Shackle, Leaf Spring Shackles and Hangers, Leak Detection Pump, Leveling Kit, License Plate Bracket, License Plate Light, License Plate Light Lens, Lift Strut, Lift Support, Liftgate Lock Actuator, Light Bar, Light Bulb, Light Guard, Locking Hub, Lower Engine Gasket Set, Lowering Kit, Lowering Springs, Lug Nut.
- M: MAP Sensor, Manual, Mass Air Flow Sensor, Mass Air Flow Sensor Boot, Mirror, Mirror Cover, Mirror Glass, Mirror Hardware, Mirror Switch, Molding Clip, Motor Mount, Motor and Transmission Mount Bushing, Mud Flaps, Muffler.
- N: Nerf Bars, Neutral Safety Switch, Nitrous System.
- O: Offroad Light, Oil Cooler, Oil Cooler Gasket Set, Oil Cooler Line, Oil Cooler Seal, Oil Dipstick, Oil Drain Plug, Oil Drain Plug Gasket, Oil Filler Cap, Oil Filter, Oil Filter Cover, Oil Filter Housing, Oil Level Sensor, Oil Line, Oil Pan, Oil Pan Gasket, Oil Pressure Switch, Oil Pump, Oil Separator, Oil Thermostat, Oxygen Sensor, Oxygen Sensor Harness.
- P: PCV Valve, PCV Valve Diaphragm, Parking Assist Sensor, Parking Brake Cable, Parking Brake Shoe, Parking Light, Performance Module, Performance Monitor, Pickup Coil, Piston, Piston Ring Set, Pitman Arm, Power Programmer, Power Steering Hose, Power Steering Pressure Hose, Power Steering Pressure Switch, Power Steering Pump, Power Steering Pump Pulley, Power Steering Reservoir, Pressure Plate, Purge Valve.
- Q: Quarter Panel, Quarter Panel Extension, Quarter Panel Molding.
- R: Rack And Pinion Assembly, Radar Detector, Radiator, Radiator Cap, Radiator Fan, Radiator Hose, Radiator Mount Bracket, Radiator Support, Radiator Support Bracket, Radiator Support Cover, Radius Arm Bushing, Rear View Mirror, Reference Sensor, Reflector, Relay, Release Bearing, Remote Starter Wiring Harness, Repair Manual, Resonator, Ring and Pinion, Rocker Panel, Rocker Panel Guards, Rocker Panel Trim, Rod Bearing Set, Roll Pan, Roof Rack, Running Board Mounting Kit, Running Boards.
- S: Seat, Seat Belt, Seat Cover, Seat Heat Pad, Seat Switch, Serpentine Belt, Shift Boot, Shift Cable, Shift Knob, Shifter, Shifter Bezel, Shifter Repair Kit, Shock Absorber and Strut Assembly, Shock And Strut Mount, Shock Bracket, Shock Bump Stop, Shock Conversion Kit, Shock and Strut Boot, Short Ram Intake, Side Marker, Side Marker Lens, Side Steps, Skid Plate, Soft Top, Spare Tire Carrier, Spark Plug, Spark Plug Wire, Speaker, Speed Sender, Speed Sensor, Speed Sensor Harness, Spindle, Splash Shield, Spoiler, Spring And Bolt Kit, Spring Seat, Starter, Steering Angle Sensor, Steering Column Bearing, Steering Coupling, Steering Damper, Steering Gearbox, Steering Knuckle, Steering Knuckle Bushing, Steering Linkage Assembly, Steering Rack, Steering Rack Boot, Steering Rack Bushing, Steering Shaft, Steering Stabilizer, Steering Wheel, Steering Wheel Installation Kit, Step Bumper, Strut Bar, Strut Bearing, Strut Insert, Strut Mount Bushing, Strut Rod Bushing, Subframe Mount, Supercharger Kit, Suspension Bushing, Suspension Kit, Suspension Lift Kit, Suspension Mount, Suspension Sensor, Sway Bar Bushing, Sway Bar Kit, Sway Bar Link, Sway Bar Link Kit, Switch.
- T: T Connector, T-Belt Tension Adjuster, T-Belt Tensioner Pulley, TPMS Sensor, Tail Light, Tail Light Circuit Board, Tail Light Connector Plate, Tail Light Cover, Tail Light Lens, Tail Pipe, Tailgate, Tailgate Cable, Tailgate Cap, Tailgate Handle, Tailgate Hinge, Tailgate Latch, Tailgate Light Bar, Tailgate Liner, Tailgate Lock, Tailgate Molding, Tailgate Protector, Temperature Sender, Thermostat, Thermostat Gasket, Thermostat Housing, Third Brake Light, Throttle Body, Throttle Body Spacer, Throttle Cable, Throttle Position Sensor, Tie Rod Adjusting Sleeve, Tie Rod Assembly, Tie Rod End, Timing Belt, Timing Belt Idler Pulley, Timing Belt Kit, Timing Belt Tensioner, Timing Chain, Timing Chain Guide, Timing Chain Kit, Timing Chain Tensioner, Timing Cover, Timing Cover Gasket, Timing Gear, Tonneau Cover, Tool Box, Tornado Fuel Saver, Torque Converter, Tow Eye Cover, Track Bar, Trailing Arm, Trailing Arm Bushing, Transfer Case, Transfer Case Gear, Transfer Case Motor, Transfer Case Shift Mode Selector, Transfer Case Switch, Transmission Control Module, Transmission Mount, Transmission Oil Cooler, Transmission Oil Line, Transmission Pan, Truck Bed Rack, Truck Tool Box, Trunk Actuator, Trunk Lid, Trunk Lid Molding, Trunk Lock, Tune Up Kit, Turbo Pipe, Turbocharger, Turbocharger Boost Solenoid, Turn Signal Cam, Turn Signal Lens, Turn Signal Light, Turn Signal Switch.
- U: U Joint, Universal Air Filter.
- V: Vacuum Pump, Vacuum Supply Pump, Vacuum Valve, Valance, Valve Cover, Valve Cover Gasket, Valve Lifter, Valve Stem Seal, Vapor Canister, Vapor Canister Check Valve, Vapor Canister Purge Solenoid, Vapor Canister Vent Solenoid, Variable Timing Solenoid, Voltage Regulator, Voltmeter.
- W: Washer Hose, Washer Pump, Washer Reservoir, Washer Reservoir Cap, Water Outlet, Water Pump, Water Pump Gasket, Water Pump Inlet Tube, Watts Link, Weatherstrip Seal, Wheel, Wheel Arch Repair Panel, Wheel Bearing, Wheel Center Cap, Wheel Cover, Wheel Cylinder, Wheel Hub, Wheelhouse, Winch, Winch Mount, Window Channel, Window Crank, Window Motor, Window Regulator, Window Switch, Window Visor, Windshield, Windshield Brackets, Windshield Frame, Windshield Hardware, Windshield Hinges, Windshield Washer Nozzle, Wiper Arm, Wiper Blade, Wiper Cowl, Wiper Cowl Grille, Wiper Linkage, Wiper Motor, Wiper Pulse Module, Wiper Switch, Wiring Harness.

Corporate & Contact Information:
Address: 761 Progress Parkway, La Salle, IL 61301 United States
Phone Numbers:
- Toll Free: 1-866-529-0412
- International: 1-216-643-6600
- Fax: 1-216-643-6610
Customer Service & Resources: Live Chat, FAQ/Help Center, Shipping Policy, Retrieve Quote, Return Policy, Feedback, Car Parts 101, Car Problem 101, Car Care 101, Driving 101, Road Tests.
"""

def ingest_doc(db, title, content):
    existing = db.query(models.BotSource).filter(models.BotSource.title == title, models.BotSource.orgId == ORG_ID).first()
    if existing:
        print(f"Deleting existing source: {title}")
        db.delete(existing)
        db.commit()

    source_id = str(uuid.uuid4())
    src = models.BotSource(
        id=source_id,
        botId=None,
        orgId=ORG_ID,
        title=title,
        kind="DOC",
        isUniversal=True,
        createdAt=datetime.datetime.utcnow()
    )
    db.add(src)
    db.commit()

    chunks = chunk_text(content, chunk_size=800, chunk_overlap=100)
    print(f"[{title}] Created {len(chunks)} chunks.")

    for i, c in enumerate(chunks):
        vec = get_embedding(c)
        chunk_rec = models.DocumentChunk(
            id=str(uuid.uuid4()),
            sourceId=source_id,
            content=f"{title}\n{c}",
            embedding=vec,
            createdAt=datetime.datetime.utcnow(),
            document_type="DOC",
            status="APPROVED",
            isFormallyReviewed=True,
            language="en"
        )
        db.add(chunk_rec)
    db.commit()
    print(f"[{title}] Ingested successfully into PostgreSQL!")

if __name__ == "__main__":
    db = SessionLocal()
    try:
        ingest_doc(db, DOC_1_TITLE, DOC_1_CONTENT)
        ingest_doc(db, DOC_2_TITLE, DOC_2_CONTENT)
        print("All documents indexed in Universal Knowledge Base successfully!")
    finally:
        db.close()
