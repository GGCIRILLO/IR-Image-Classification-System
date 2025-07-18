"""
Object class definitions for IR image classification system.
Each class represents a specific object type found in the data/raw2.0 folder.
"""

from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass


class ObjectCategory(Enum):
    """High-level categories for object classification."""
    MILITARY_VEHICLE = "military_vehicle"
    CIVILIAN_VEHICLE = "civilian_vehicle"
    AIR_DEFENSE = "air_defense"
    MISSILE_SYSTEM = "missile_system"
    BUILDING = "building"
    INFRASTRUCTURE = "infrastructure"
    LAUNCH_PAD = "launch_pad"
    COMMUNICATION = "communication"


@dataclass
class ObjectClass:
    """Represents a single object class with metadata."""
    name: str
    folder_name: str
    category: ObjectCategory
    description: Optional[str] = None
    aliases: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []


class ObjectClassRegistry:
    """Registry for all object classes in the dataset."""
    
    def __init__(self):
        self._classes = self._initialize_classes()
        self._name_to_class = {cls.name: cls for cls in self._classes}
        self._folder_to_class = {cls.folder_name: cls for cls in self._classes}
    
    def _initialize_classes(self) -> List[ObjectClass]:
        """Initialize all object classes based on folder structure."""
        return [
            # Military Vehicles - Tanks
            ObjectClass("AAV_Tank", "AAV Tank", ObjectCategory.MILITARY_VEHICLE, 
                       "Amphibious Assault Vehicle Tank"),
            ObjectClass("BMP1_APC_Tank", "BMP-1 APC Tank", ObjectCategory.MILITARY_VEHICLE,
                       "Soviet infantry fighting vehicle"),
            ObjectClass("BMP2_APC_Tank", "BMP-2 APC Tank", ObjectCategory.MILITARY_VEHICLE,
                       "Soviet infantry fighting vehicle, improved version"),
            ObjectClass("BTR80_APC", "BTR-80 APC", ObjectCategory.MILITARY_VEHICLE,
                       "Soviet armored personnel carrier"),
            ObjectClass("Challenger_Tank", "Challenger Tank", ObjectCategory.MILITARY_VEHICLE,
                       "British main battle tank"),
            ObjectClass("M1A1_Abrams_Tank", "M-1A1 Abrams Tank", ObjectCategory.MILITARY_VEHICLE,
                       "American main battle tank"),
            ObjectClass("M2_Bradley_APC_Tank", "M2 Bradley Apc Tank", ObjectCategory.MILITARY_VEHICLE,
                       "American infantry fighting vehicle"),
            ObjectClass("Mercava", "Mercava", ObjectCategory.MILITARY_VEHICLE,
                       "Israeli main battle tank"),
            ObjectClass("T54_Tank", "T54 Tank", ObjectCategory.MILITARY_VEHICLE,
                       "Soviet medium tank"),
            ObjectClass("T62_Tank", "T62 Tank", ObjectCategory.MILITARY_VEHICLE,
                       "Soviet main battle tank"),
            ObjectClass("T64_Tank", "T64 Tank", ObjectCategory.MILITARY_VEHICLE,
                       "Soviet main battle tank"),
            ObjectClass("T72_Tank", "T72 Tank", ObjectCategory.MILITARY_VEHICLE,
                       "Soviet main battle tank"),
            ObjectClass("T80_Tank", "T80 Tank", ObjectCategory.MILITARY_VEHICLE,
                       "Soviet main battle tank"),
            ObjectClass("Warrior_AFV_Tank", "Warrior AFV Tank", ObjectCategory.MILITARY_VEHICLE,
                       "British armored fighting vehicle"),
            ObjectClass("ZSU_23_4_Anti_Aircraft", "ZSU-23-4 Anti Aircraft Artillery Tank", ObjectCategory.MILITARY_VEHICLE,
                       "Soviet self-propelled anti-aircraft gun"),
            
            # Military Vehicles - Other
            ObjectClass("Armored_Car", "Armored Car", ObjectCategory.MILITARY_VEHICLE,
                       "Light armored reconnaissance vehicle"),
            ObjectClass("BRDM_AVR", "BRDM AVR", ObjectCategory.MILITARY_VEHICLE,
                       "Soviet armored reconnaissance vehicle"),
            ObjectClass("Humvee", "Humvee", ObjectCategory.MILITARY_VEHICLE,
                       "High Mobility Multipurpose Wheeled Vehicle"),
            ObjectClass("Humvee_with_Stingers", "Humvee with Stingers", ObjectCategory.MILITARY_VEHICLE,
                       "Humvee equipped with Stinger missiles"),
            ObjectClass("M109A3_Self_Propelled_Gun", "M-109A3 Self Propelled Gun", ObjectCategory.MILITARY_VEHICLE,
                       "American self-propelled howitzer"),
            ObjectClass("M981_FISTV", "M-981 FISTV", ObjectCategory.MILITARY_VEHICLE,
                       "Fire Support Team Vehicle"),
            ObjectClass("M981_Mobile_Air_Defense", "M-981 Mobile Air Defense", ObjectCategory.MILITARY_VEHICLE,
                       "Mobile air defense system"),
            ObjectClass("M163_Vulcan", "M163 Vulcan", ObjectCategory.MILITARY_VEHICLE,
                       "Self-propelled anti-aircraft gun"),
            ObjectClass("M35A_Truck", "M35-A Truck", ObjectCategory.MILITARY_VEHICLE,
                       "Military cargo truck"),
            
            # Civilian Vehicles
            ObjectClass("Fuel_Truck", "Fuel Truck", ObjectCategory.CIVILIAN_VEHICLE,
                       "Commercial fuel transport vehicle"),
            ObjectClass("Jeep", "Jeep", ObjectCategory.CIVILIAN_VEHICLE,
                       "Light utility vehicle"),
            ObjectClass("PickUp", "PickUp", ObjectCategory.CIVILIAN_VEHICLE,
                       "Pickup truck"),
            ObjectClass("Semi_Tractor", "Semi Tractor", ObjectCategory.CIVILIAN_VEHICLE,
                       "Semi-trailer truck tractor"),
            ObjectClass("Semi_Tractor_Tanker", "Semi Tractor Tanker", ObjectCategory.CIVILIAN_VEHICLE,
                       "Semi-trailer tanker truck"),
            ObjectClass("Semi_Tractor_Trailer", "Semi Tractor Trailer", ObjectCategory.CIVILIAN_VEHICLE,
                       "Semi-trailer truck with trailer"),
            
            # Air Defense Systems
            ObjectClass("Hawk_Air_Defense", "Hawk Air Defense", ObjectCategory.AIR_DEFENSE,
                       "Medium-range surface-to-air missile system"),
            ObjectClass("Patriot", "Patriot", ObjectCategory.AIR_DEFENSE,
                       "Surface-to-air missile system"),
            ObjectClass("Patriot_Air_Defense_Launcher", "Patriot Air Defense Launcher", ObjectCategory.AIR_DEFENSE,
                       "Patriot missile launcher unit"),
            ObjectClass("Patriot_Air_Defense_Radar", "Patriot Air Defense Radar", ObjectCategory.AIR_DEFENSE,
                       "Patriot radar system"),
            ObjectClass("Patriot_Antennas", "Patriot Antennas", ObjectCategory.AIR_DEFENSE,
                       "Patriot communication antennas"),
            ObjectClass("Patriot_Control", "Patriot Control", ObjectCategory.AIR_DEFENSE,
                       "Patriot control station"),
            ObjectClass("Patriot_Missile_Truck", "Patriot Missile Truck", ObjectCategory.AIR_DEFENSE,
                       "Patriot missile transport vehicle"),
            ObjectClass("Patriot_Radar", "Patriot Radar", ObjectCategory.AIR_DEFENSE,
                       "Patriot radar component"),
            ObjectClass("Rapier_Air_Defense_Radar", "Rapier Air Defense Radar", ObjectCategory.AIR_DEFENSE,
                       "British surface-to-air missile radar"),
            ObjectClass("SA10_Air_Defense_System", "SA-10 Air Defense System", ObjectCategory.AIR_DEFENSE,
                       "Soviet/Russian surface-to-air missile system"),
            ObjectClass("SA3_Low_Blow_Radar", "SA-3 Low Blow Radar", ObjectCategory.AIR_DEFENSE,
                       "Soviet surface-to-air missile radar"),
            ObjectClass("SA4_Air_Defense_System", "SA-4 Air Defense System", ObjectCategory.AIR_DEFENSE,
                       "Soviet surface-to-air missile system"),
            ObjectClass("SA6_Air_Defense_System", "SA-6 Air Defense System", ObjectCategory.AIR_DEFENSE,
                       "Soviet surface-to-air missile system"),
            ObjectClass("SA8_Air_Defense_System", "SA-8 Air Defense System", ObjectCategory.AIR_DEFENSE,
                       "Soviet surface-to-air missile system"),
            ObjectClass("THAAD_Launcher", "THAAD Launcher", ObjectCategory.AIR_DEFENSE,
                       "Terminal High Altitude Area Defense launcher"),
            ObjectClass("THAAD_Radar", "THAAD Radar", ObjectCategory.AIR_DEFENSE,
                       "THAAD radar system"),           
 
            # Missile Systems
            ObjectClass("BGM109_Tomahawk_Cruise", "BGM-109 Tomahawk Cruise", ObjectCategory.MISSILE_SYSTEM,
                       "Subsonic cruise missile"),
            ObjectClass("MAZ543_Scud_TEL", "MAZ543 Scud TEL", ObjectCategory.MISSILE_SYSTEM,
                       "Transporter Erector Launcher for Scud missiles"),
            ObjectClass("MAZ543_Scud_Tel_Launcher", "MAZ543 Scud Tel Launcher", ObjectCategory.MISSILE_SYSTEM,
                       "Scud missile launcher system"),
            ObjectClass("Scud_Tel", "Scud Tel", ObjectCategory.MISSILE_SYSTEM,
                       "Scud missile transporter erector launcher"),
            
            # Launch Pads
            ObjectClass("Ariane_LP", "Ariane-LP", ObjectCategory.LAUNCH_PAD,
                       "Ariane rocket launch pad"),
            ObjectClass("Athena2_LP", "Athena2-LP", ObjectCategory.LAUNCH_PAD,
                       "Athena II rocket launch pad"),
            ObjectClass("Atlas_LP", "Atlas LP", ObjectCategory.LAUNCH_PAD,
                       "Atlas rocket launch pad"),
            ObjectClass("SLC_4E_Pad", "SLC 4E Pad", ObjectCategory.LAUNCH_PAD,
                       "Space Launch Complex 4E pad"),
            ObjectClass("Taurus_LP", "Taurus LP", ObjectCategory.LAUNCH_PAD,
                       "Taurus rocket launch pad"),
            ObjectClass("Titan4B_LP", "Titan4B LP", ObjectCategory.LAUNCH_PAD,
                       "Titan IVB rocket launch pad"),
            
            # Buildings
            ObjectClass("Airfield", "Airfield", ObjectCategory.BUILDING,
                       "Airport or military airfield"),
            ObjectClass("Bakery", "Bakery", ObjectCategory.BUILDING,
                       "Commercial bakery building"),
            ObjectClass("Bank", "Bank", ObjectCategory.BUILDING,
                       "Financial institution building"),
            ObjectClass("Barn", "Barn", ObjectCategory.BUILDING,
                       "Agricultural storage building"),
            ObjectClass("Bowling_Alley", "Bowling Alley", ObjectCategory.BUILDING,
                       "Recreation facility"),
            ObjectClass("Building", "Building", ObjectCategory.BUILDING,
                       "Generic building structure"),
            ObjectClass("Camp", "Camp", ObjectCategory.BUILDING,
                       "Military or civilian camp facility"),
            ObjectClass("Car_Dealership", "Car dealership", ObjectCategory.BUILDING,
                       "Automotive sales facility"),
            ObjectClass("Factory", "Factory", ObjectCategory.BUILDING,
                       "Industrial manufacturing facility"),
            ObjectClass("Military_Complex", "Military Complex", ObjectCategory.BUILDING,
                       "Military installation or base"),
            ObjectClass("Silo", "Silo", ObjectCategory.BUILDING,
                       "Storage silo structure"),
            
            # Infrastructure
            ObjectClass("Bridge", "Bridge", ObjectCategory.INFRASTRUCTURE,
                       "Transportation bridge structure"),
            ObjectClass("Facility_Hex", "Facility Hex", ObjectCategory.INFRASTRUCTURE,
                       "Hexagonal facility structure"),
            ObjectClass("Fossil_Fuel_Power_Plant", "Fossil Fuel Power Plant", ObjectCategory.INFRASTRUCTURE,
                       "Coal or gas power generation facility"),
            ObjectClass("LargeOil_Tank", "LargeOil Tank", ObjectCategory.INFRASTRUCTURE,
                       "Large petroleum storage tank"),
            ObjectClass("LightHouse", "LightHouse", ObjectCategory.INFRASTRUCTURE,
                       "Maritime navigation aid"),
            ObjectClass("Power_Tower", "Power Tower", ObjectCategory.INFRASTRUCTURE,
                       "Electrical transmission tower"),
            ObjectClass("Water_Tower", "Water Tower", ObjectCategory.INFRASTRUCTURE,
                       "Water storage and distribution tower"),
            
            # Communication Systems
            ObjectClass("MSE_Terminal", "MSE Terminal", ObjectCategory.COMMUNICATION,
                       "Mobile Subscriber Equipment terminal"),
            ObjectClass("SatDish", "SatDish", ObjectCategory.COMMUNICATION,
                       "Satellite communication dish"),
            ObjectClass("Satellite_Ground_Station", "Satellite Ground Station", ObjectCategory.COMMUNICATION,
                       "Satellite communication ground station"),
            ObjectClass("Smart_Tee", "Smart Tee", ObjectCategory.COMMUNICATION,
                       "Communication equipment"),
            ObjectClass("TSC_100_Comm_Terminal", "TSC 100 Comm Terminal", ObjectCategory.COMMUNICATION,
                       "Tactical satellite communication terminal"),
            ObjectClass("TSC_85B_CSCS_Comm_Terminal", "TSC 85B CSCS Comm Terminal", ObjectCategory.COMMUNICATION,
                       "Tactical communication terminal"),
            ObjectClass("Tnshdrad", "Tnshdrad", ObjectCategory.COMMUNICATION,
                       "Communication radar system"),
            
            # Specialized Equipment
            ObjectClass("Scamp", "Scamp", ObjectCategory.MILITARY_VEHICLE,
                       "Specialized military equipment"),
        ]
    
    def get_class_by_name(self, name: str) -> Optional[ObjectClass]:
        """Get object class by standardized name."""
        return self._name_to_class.get(name)
    
    def get_class_by_folder(self, folder_name: str) -> Optional[ObjectClass]:
        """Get object class by folder name."""
        return self._folder_to_class.get(folder_name)
    
    def get_classes_by_category(self, category: ObjectCategory) -> List[ObjectClass]:
        """Get all classes in a specific category."""
        return [cls for cls in self._classes if cls.category == category]
    
    def get_all_classes(self) -> List[ObjectClass]:
        """Get all object classes."""
        return self._classes.copy()
    
    def get_class_names(self) -> List[str]:
        """Get list of all standardized class names."""
        return list(self._name_to_class.keys())
    
    def get_folder_names(self) -> List[str]:
        """Get list of all folder names."""
        return list(self._folder_to_class.keys())
    
    def get_category_distribution(self) -> Dict[ObjectCategory, int]:
        """Get count of classes per category."""
        distribution = {}
        for cls in self._classes:
            distribution[cls.category] = distribution.get(cls.category, 0) + 1
        return distribution
    
    def __len__(self) -> int:
        """Return total number of classes."""
        return len(self._classes)
    
    def __iter__(self):
        """Iterate over all classes."""
        return iter(self._classes)


# Global registry instance
OBJECT_REGISTRY = ObjectClassRegistry()


def get_object_classes() -> ObjectClassRegistry:
    """Get the global object class registry."""
    return OBJECT_REGISTRY


def get_class_id_mapping() -> Dict[str, int]:
    """Get mapping from class names to numeric IDs for model training."""
    return {name: idx for idx, name in enumerate(OBJECT_REGISTRY.get_class_names())}


def get_id_class_mapping() -> Dict[int, str]:
    """Get mapping from numeric IDs to class names for model inference."""
    return {idx: name for name, idx in get_class_id_mapping().items()}