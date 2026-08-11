"""
seed_content.py — Synthetic NCERT-style content for testing the RAG engine.
"""

from typing import List
from rag.models import ConceptContent

def get_seed_content() -> List[ConceptContent]:
    """Return synthetic NCERT Class 9 textbook content for selected concepts."""
    
    return [
        ConceptContent(
            concept_id="E01",
            concept_name="Electric Charge",
            chapter="Electricity",
            sections=[
                "Electric charge is a fundamental property of matter. It exists in two forms: positive and negative. Protons carry a positive charge, while electrons carry a negative charge. Like charges repel each other, and unlike charges attract each other.",
                "Charge is quantized, meaning it only exists in discrete amounts. The smallest unit of charge is the charge of an electron, which is approximately 1.6 x 10^-19 Coulombs. The SI unit of electric charge is the Coulomb (C).",
                "Charge is always conserved in any physical or chemical reaction. It can neither be created nor destroyed, only transferred from one body to another."
            ]
        ),
        ConceptContent(
            concept_id="E02",
            concept_name="Electric Current",
            chapter="Electricity",
            sections=[
                "Electric current is defined as the rate of flow of electric charge through any cross-section of a conductor. If a net charge Q flows across any cross-section of a conductor in time t, then the current I is given by the formula I = Q/t.",
                "The SI unit of electric current is Ampere (A). One Ampere is defined as the flow of one Coulomb of charge per second (1 A = 1 C / 1 s). Small quantities of current are expressed in milliamperes (1 mA = 10^-3 A) or microamperes (1 µA = 10^-6 A)."
            ]
        ),
        ConceptContent(
            concept_id="E05",
            concept_name="Conventional Direction of Current",
            chapter="Electricity",
            sections=[
                "By convention, the direction of electric current is taken as opposite to the direction of the flow of electrons. When electricity was first studied, electrons were not known, so current was thought to be the flow of positive charges.",
                "Therefore, in an electric circuit, conventional current flows from the positive terminal of the battery to the negative terminal. Actual electrons, being negatively charged, flow from the negative terminal to the positive terminal."
            ]
        ),
        ConceptContent(
            concept_id="E06",
            concept_name="Electric Circuit",
            chapter="Electricity",
            sections=[
                "An electric circuit is a continuous and closed path along which an electric current flows. It typically consists of a power source (like a battery), conducting wires, and one or more electrical components (like a bulb or resistor).",
                "If the circuit is broken anywhere (for example, if a switch is turned off or a wire breaks), the current stops flowing immediately. This is known as an open circuit."
            ]
        ),
        ConceptContent(
            concept_id="E07",
            concept_name="Ammeter",
            chapter="Electricity",
            sections=[
                "An ammeter is an instrument used to measure the electric current flowing in a circuit. It is always connected in series with the component through which the current is to be measured.",
                "Because it is connected in series, an ammeter must have a very low electrical resistance so that it does not significantly alter the current it is trying to measure."
            ]
        ),
        ConceptContent(
            concept_id="E10",
            concept_name="Voltmeter",
            chapter="Electricity",
            sections=[
                "A voltmeter is an instrument used to measure the potential difference (voltage) between two points in an electric circuit. It is always connected in parallel across the points where the potential difference is to be measured.",
                "To ensure it draws negligible current from the circuit and does not disturb the voltage being measured, a voltmeter must have a very high electrical resistance."
            ]
        ),
        ConceptContent(
            concept_id="E13",
            concept_name="Ohm's Law",
            chapter="Electricity",
            sections=[
                "Ohm's Law states that the potential difference (V) across the ends of a given metallic wire in an electric circuit is directly proportional to the current (I) flowing through it, provided its temperature remains the same.",
                "This relationship is expressed mathematically as V ∝ I, or V = IR, where R is a constant called the resistance of the conductor. For ohmic conductors, the V-I graph is a straight line passing through the origin."
            ]
        ),
        ConceptContent(
            concept_id="E16",
            concept_name="V-I Graph",
            chapter="Electricity",
            sections=[
                "Resistance is the property of a conductor to resist the flow of charges through it. Its SI unit is the ohm (Ω). According to Ohm's Law, R = V/I.",
                "The resistance of a conductor depends on its length (l), its cross-sectional area (A), and the nature of its material. It is given by R = ρ(l/A), where ρ (rho) is the electrical resistivity of the material."
            ]
        ),
        ConceptContent(
            concept_id="E14",
            concept_name="Resistance",
            chapter="Electricity",
            sections=[
                "Resistance is a property that resists the flow of electrons in a conductor. It controls the magnitude of the current. The SI unit of resistance is the ohm (Ω).",
                "If the potential difference across the two ends of a conductor is 1 V and the current through it is 1 A, then the resistance R, of the conductor is 1 Ω. That is, 1 Ω = 1 V / 1 A."
            ]
        ),
        ConceptContent(
            concept_id="E36",
            concept_name="Joule's Law of Heating",
            chapter="Electricity",
            sections=[
                "The heat produced in a resistor is (i) directly proportional to the square of current for a given resistance, (ii) directly proportional to resistance for a given current, and (iii) directly proportional to the time for which the current flows through the resistor.",
                "This is known as Joule's law of heating. The law implies that heat produced in a resistor is given by H = I^2 R t."
            ]
        ),
        ConceptContent(
            concept_id="E40",
            concept_name="Electric Power",
            chapter="Electricity",
            sections=[
                "The rate at which electric energy is dissipated or consumed in an electric circuit is termed as electric power. The power P is given by P = VI.",
                "Or P = I^2 R = V^2 / R. The SI unit of electric power is watt (W). It is the power consumed by a device that carries 1 A of current when operated at a potential difference of 1 V."
            ]
        ),
        ConceptContent(
            concept_id="M03",
            concept_name="Properties of Magnetic Field Lines",
            chapter="Magnetic Effects of Electric Current",
            sections=[
                "Magnetic field lines are continuous closed curves. They emerge from the North pole and merge at the South pole. Inside the magnet, the direction of field lines is from its South pole to its North pole.",
                "No two field lines are found to cross each other. If they did, it would mean that at the point of intersection, the compass needle would point towards two directions, which is not possible."
            ]
        ),
        ConceptContent(
            concept_id="M09",
            concept_name="Oersted's Experiment",
            chapter="Magnetic Effects of Electric Current",
            sections=[
                "Hans Christian Oersted showed that electricity and magnetism are related phenomena. He observed that a compass needle suffered a deflection when placed near a wire carrying an electric current.",
                "The deflection of the compass needle reverses if the direction of current in the wire is reversed, proving that the magnetic field direction depends on the direction of current."
            ]
        ),
        ConceptContent(
            concept_id="M12",
            concept_name="Right-Hand Thumb Rule",
            chapter="Magnetic Effects of Electric Current",
            sections=[
                "A convenient way of finding the direction of magnetic field associated with a current-carrying conductor is given by the right-hand thumb rule.",
                "Imagine that you are holding a current-carrying straight conductor in your right hand such that the thumb points towards the direction of current. Then your fingers will wrap around the conductor in the direction of the field lines of the magnetic field."
            ]
        ),
        ConceptContent(
            concept_id="M15",
            concept_name="Solenoid",
            chapter="Magnetic Effects of Electric Current",
            sections=[
                "A coil of many circular turns of insulated copper wire wrapped closely in the shape of a cylinder is called a solenoid.",
                "The pattern of the magnetic field lines around a current-carrying solenoid is similar to that of a bar magnet. The field lines inside the solenoid are in the form of parallel straight lines, indicating that the magnetic field is the same at all points inside the solenoid."
            ]
        ),
        ConceptContent(
            concept_id="M31",
            concept_name="Electric Motor — Principle",
            chapter="Magnetic Effects of Electric Current",
            sections=[
                "An electric motor is a rotating device that converts electrical energy into mechanical energy.",
                "It works on the principle that a current-carrying conductor placed in a magnetic field experiences a force. The direction of this force is given by Fleming's Left-Hand Rule."
            ]
        ),
        ConceptContent(
            concept_id="M33",
            concept_name="Electromagnetic Induction",
            chapter="Magnetic Effects of Electric Current",
            sections=[
                "The process by which a changing magnetic field in a conductor induces a current in another conductor is called electromagnetic induction.",
                "This phenomenon was discovered by Michael Faraday. If a coil is moved within a magnetic field, or if the magnetic field around a coil is changed, an induced potential difference is set up."
            ]
        )
    ]
