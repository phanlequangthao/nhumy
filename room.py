import uuid
from datetime import datetime
from typing import Dict, List

class Room:
    def __init__(self, room_id: str, host_name: str):
        self.room_id = room_id
        self.host_name = host_name
        self.created_at = datetime.now()
        self.participants: Dict[str, str] = {}  # {participant_id: name}
        self.is_active = True

    def add_participant(self, participant_id: str, name: str):
        self.participants[participant_id] = name

    def remove_participant(self, participant_id: str):
        if participant_id in self.participants:
            del self.participants[participant_id]

    def get_participants(self) -> List[str]:
        return list(self.participants.values())

class RoomManager:
    def __init__(self):
        self.rooms: Dict[str, Room] = {}

    def create_room(self, host_name: str) -> str:
        room_id = str(uuid.uuid4())[:8]  # Tạo ID ngắn gọn
        self.rooms[room_id] = Room(room_id, host_name)
        return room_id

    def get_room(self, room_id: str) -> Room:
        return self.rooms.get(room_id)

    def remove_room(self, room_id: str):
        if room_id in self.rooms:
            del self.rooms[room_id]

    def list_active_rooms(self) -> List[Dict]:
        return [
            {
                "room_id": room_id,
                "host": room.host_name,
                "participants": len(room.participants),
                "created_at": room.created_at
            }
            for room_id, room in self.rooms.items()
            if room.is_active
        ]

# Tạo instance global của RoomManager
room_manager = RoomManager() 