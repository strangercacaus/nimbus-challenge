from typing import Dict, Any, List
from utils import async_retry, get_notion_page_name_from_dw, get_notion_username_from_dw
from data_classes import NotionPage
from notion_client import AsyncClient
import logging

logger = logging.getLogger(__name__)

class NotionConnector:
    def __init__(self, notion_api_key: str):
        self.notion_api_key = notion_api_key
        self.notion_client = AsyncClient(auth=notion_api_key)

    @async_retry()
    async def get_notion_page_blocks(self, page_id: str) -> List[Dict[str, Any]]:
        """Fetch all blocks from a Notion page recursively."""
        if self.notion_client is None:
            raise Exception("Notion client not initialized")
        
        all_blocks = []
        
        async def fetch_blocks_recursive(block_id: str, parent_type: str = "page"):
            """Recursively fetch all blocks and their children."""
            try:
                if parent_type == "page":
                    response = await self.notion_client.blocks.children.list(block_id=block_id)
                else:
                    response = await self.notion_client.blocks.children.list(block_id=block_id)
                
                blocks = response.get("results", [])
                
                for block in blocks:
                    all_blocks.append(block)
                    
                    # If block has children, fetch them recursively
                    if block.get("has_children", False):
                        await fetch_blocks_recursive(block["id"], "block")
                        
            except Exception as e:
                logger.info(f"Error fetching blocks for {block_id}: {e}")
        
        await fetch_blocks_recursive(page_id, "page")
        return all_blocks
    
    @async_retry()
    async def get_notion_page_comments(self, page_id: str) -> List[Dict[str, Any]]:
        """Fetch all comments from a Notion page."""
        if self.notion_client is None:
            raise Exception("Notion client not initialized")
        
        all_comments = []
        
        try:
            # Fetch comments for the page
            response = await self.notion_client.comments.list(block_id=page_id)
            comments = response.get("results", [])
            
            for comment in comments:
                all_comments.append(comment)
                
        except Exception as e:
            logger.info(f"Error fetching comments for {page_id}: {e}")
        
        return all_comments
    
    @async_retry()
    async def get_notion_page_properties(self, page_id: str) -> List[Dict[str, Any]]:
        """Fetch all properties from a Notion page and return as list of dicts."""
        if self.notion_client is None:
            raise Exception("Notion client not initialized")
        
        try:
            # Fetch the page to get its properties
            response = await self.notion_client.pages.retrieve(page_id=page_id)
            properties = response.get("properties", {})
            
            # Transform properties dict to list of dicts
            properties_list = []
            for prop_name, prop_value in properties.items():
                prop_dict = {
                    "name": prop_name,
                    "type": prop_value.get("type"),
                    "value": prop_value.get(prop_value.get("type"))
                }
                properties_list.append(prop_dict)
            
            return properties_list
                
        except Exception as e:
            logger.info(f"Error fetching properties for {page_id}: {e}")
            return []

    def notion_block_to_text(self, block: Dict[str, Any]) -> str:
        """Convert a Notion block to text format."""
        block_type = block.get("type", "")
        
        if block_type == "paragraph":
            return self.notion_rich_text_to_string(block["paragraph"]["rich_text"])
        
        elif block_type == "heading_1":
            text = self.notion_rich_text_to_string(block["heading_1"]["rich_text"])
            return f"# {text}\n"
        
        elif block_type == "heading_2":
            text = self.notion_rich_text_to_string(block["heading_2"]["rich_text"])
            return f"## {text}\n"
        
        elif block_type == "heading_3":
            text = self.notion_rich_text_to_string(block["heading_3"]["rich_text"])
            return f"### {text}\n"
        
        elif block_type == "bulleted_list_item":
            text = self.notion_rich_text_to_string(block["bulleted_list_item"]["rich_text"])
            return f"{text}"
        
        elif block_type == "numbered_list_item":
            text = self.notion_rich_text_to_string(block["numbered_list_item"]["rich_text"])
            return f"{text}"
        
        elif block_type == "to_do":
            text = self.notion_rich_text_to_string(block["to_do"]["rich_text"])
            checked = "Feito" if block["to_do"]["checked"] else "A Fazer"
            return f"{checked}: {text}"
        
        elif block_type == "toggle":
            text = self.notion_rich_text_to_string(block["toggle"]["rich_text"])
            return f"{text}"
        
        elif block_type == "code":
            language = block["code"].get("language", "")
            text = self.notion_rich_text_to_string(block["code"]["rich_text"])
            return f"{language}\n{text}\n"
        
        elif block_type == "quote":
            text = self.notion_rich_text_to_string(block["quote"]["rich_text"])
            return f"{text}"
        
        elif block_type == "callout":
            # icon = block["callout"].get("icon", {})
            # if icon.get("type") == "emoji":
            #     emoji = icon.get("emoji", "")
            # else:
            #     emoji = "[!]"
            text = self.notion_rich_text_to_string(block["callout"]["rich_text"])
            return f"{text}"
        
        # elif block_type == "divider":
        #     return "---"
        
        # elif block_type == "table":
            # For tables, we'll return a simple indicator since actual table structure
            # would need more complex handling
            # return "[Table content]"
        
        elif block_type == "table_row":
            cells = block["table_row"]["cells"]
            row_text = ", ".join([self.notion_rich_text_to_string(cell) for cell in cells])
            return f"{row_text}"
        
        elif block_type in ["image", "video", "file", "pdf"]:
            # For media blocks, return URL and description
            url = ""
            caption = ""

            if block_type == "image":
                image_data = block["image"]
                # # Get URL based on type
                # if image_data.get("type") == "external":
                #     url = image_data.get("external", {}).get("url", "")
                # elif image_data.get("type") == "file":
                #     url = image_data.get("file", {}).get("url", "")
                # Get caption
                if "caption" in image_data:
                    caption = self.notion_rich_text_to_string(image_data["caption"])

            elif block_type == "video":
                video_data = block["video"]
                # if video_data.get("type") == "external":
                #     url = video_data.get("external", {}).get("url", "")
                # elif video_data.get("type") == "file":
                #     url = video_data.get("file", {}).get("url", "")
                if "caption" in video_data:
                    caption = self.notion_rich_text_to_string(video_data["caption"])

            elif block_type == "file":
                file_data = block["file"]
                # if file_data.get("type") == "external":
                #     url = file_data.get("external", {}).get("url", "")
                # elif file_data.get("type") == "file":
                #     url = file_data.get("file", {}).get("url", "")
                if "caption" in file_data:
                    caption = self.notion_rich_text_to_string(file_data["caption"])

            # Build return string with URL and caption
            parts = []
            if len(caption) > 5:
                parts.append(f"A legenda de um {block_type} é {caption}")

            if len(parts) > 0:
                return f"{','.join(parts)}"
        
        elif block_type == "bookmark":
            url = block["bookmark"]["url"]
            caption = self.notion_rich_text_to_string(block["bookmark"].get("caption", []))
            return f"[Bookmark: {url}]" + (f" - {caption}" if caption else "")
        
        elif block_type == "link_preview":
            url = block["link_preview"]["url"]
            return f"[Link: {url}]"
        
        elif block_type == "embed":
            url = block["embed"]["url"]
            caption = self.notion_rich_text_to_string(block["embed"].get("caption", []))
            return f"[Embed: {url}]" + (f" - {caption}" if caption else "")
        
        else:
            # For unknown block types, try to extract any rich_text content
            if block_type in block and isinstance(block[block_type], dict) and "rich_text" in block[block_type]:
                return self.notion_rich_text_to_string(block[block_type]["rich_text"])
            return None #f"${block_type}"
    
    def notion_comment_to_text(self, comment: Dict[str, Any]) -> str:
        """Convert a Notion comment to text format."""
        
        # Extract comment content
        rich_text = comment.get("rich_text", [])
        content = self.notion_rich_text_to_string(rich_text)
        
        # Extract metadata
        created_time = comment.get("created_time", "")
        created_by = comment.get("created_by", {})
        comment_id = comment.get("id", "")
        discussion_id = comment.get("discussion_id", "")
        parent_id = comment.get("parent", {}).get("page_id", "") if comment.get("parent") else ""
        
        # Extract author information
        author_name = "Um Usuário"
        if created_by:
            if created_by.get("name"):
                author_name = created_by["name"]
            elif created_by.get("id"):
                user_name = get_notion_username_from_dw(created_by["id"])
                author_name = f"{user_name}"
        
        # Format timestamp (make it more readable)
        # formatted_time = created_time
        # if created_time:
        #     try:
        #         # Extract just the date and time part, remove timezone info for readability
        #         if "T" in created_time:
        #             date_part, time_part = created_time.split("T")
        #             time_part = time_part.split(".")[0]  # Remove milliseconds
        #             formatted_time = f"{date_part} {time_part}"
        #     except:
        #         formatted_time = created_time
        
        # Build the comment text
        comment_parts = []
        
        # Header with author and time
        header = f"{author_name} comentou:"
        # if formatted_time:
        #     header += f": {formatted_time}]"
        comment_parts.append(header)
        
        # Comment content
        if content.strip():
            comment_parts.append(content)
        # else:
        #     comment_parts.append("vazio")
        
        # Optional metadata (for debugging or detailed views)
        # metadata_parts = []
        # if comment_id:
        #     metadata_parts.append(f"ID: {comment_id[:8]}")
        # if discussion_id:
        #     metadata_parts.append(f"Discussion: {discussion_id[:8]}")
        
        # if metadata_parts:
            # comment_parts.append(f"*({', '.join(metadata_parts)})*")
        
        # Join with line breaks
        return "\n"+"\n".join(comment_parts)
    
    def notion_property_to_text(self,property_dict: dict) -> str:

        # Check if it's the new format (has 'name', 'type', 'value' keys)
        if 'name' in property_dict and 'type' in property_dict and 'value' in property_dict:
            property_name = property_dict['name']
            property_type = property_dict['type']
            property_value = property_dict['value']
        else:
            # Handle old format
            property_name = list(property_dict.keys())[0]
            property_data = property_dict[property_name]
            if type(property_data) == str:
                return f"{property_name}: {property_data}" if property_data else None
            elif type(property_data) == dict:
                property_type = property_data.get("type", "")
                # Convert old format to new format for processing
                property_value = property_data.get(property_type, None)
            else:
                return None
        
        
        if property_type == "title":
            text = self.notion_rich_text_to_string(property_value) if property_value else None
            return f"{property_name} é {text}" if text.strip() else None
        
        elif property_type == "rich_text" and len(property_value) > 0:
            text = self.notion_rich_text_to_string(property_value).strip()
            return f"{property_name} é {text}" if text else None
        
        # elif property_type == "number":
        #     return f"{property_name}: {property_value}" if property_value is not None else None
        
        elif property_type == "select":
            if property_value and isinstance(property_value, dict):
                return f"{property_name} é {property_value['name']}"
            return None
        
        elif property_type == "multi_select":
            if property_value and isinstance(property_value, list):
                names = [option["name"] for option in property_value if option.get("name")]
                return f"{property_name} são {', '.join(names)}" if names else None
            return None
        
        # elif property_type == "date":
        #     if property_value and isinstance(property_value, dict):
        #         start = property_value.get("start", "")
        #         end = property_value.get("end", "")
        #         if end:
        #             return f"{property_name}: {start} → {end}"
        #         return f"{property_name}: {start}" if start else None
        #     return None
        
        elif property_type == "people":
            if property_value and isinstance(property_value, list):
                names = []
                for person in property_value:
                    if person.get("name"):
                        names.append(person["name"])
                    elif person.get("id"):
                        user_name = get_notion_username_from_dw(person["id"])
                        names.append(f"User:{user_name}")
                return f"{property_name} é {', '.join(names)}" if names else None
            return None
        
        # elif property_type == "checkbox":
        #     status = "v" if property_value else "f"
        #     return f"{property_name}: {status}"
        
        # elif property_type == "url":
        #     return f"{property_name}: {property_value}" if property_value else None
        
        elif property_type == "email":
            return f"{property_name} é {property_value}" if property_value else None
        
        # elif property_type == "phone_number":
        #     return f"{property_name}: {property_value}" if property_value else None
        
        # elif property_type == "files":
        #     if property_value and isinstance(property_value, list):
        #         file_names = []
        #         for file in property_value:
        #             if file.get("name"):
        #                 file_names.append(file["name"])
        #             elif file.get("type") == "external":
        #                 file_names.append(f"External: {file['external']['url']} ")
        #             elif file.get("type") == "file":
        #                 file_names.append(f"File: {file['file']['url']} ")
        #         return f"{property_name}: {', '.join(file_names)} " if file_names else None
        #     return None
        
        elif property_type == "relation":
            if property_value and isinstance(property_value, list):
                if len(property_value) > 0:
                    relation_names = ", ".join([get_notion_page_name_from_dw(page["id"]) for page in property_value])
                    return f"{property_name} é {relation_names} " if len(property_value) > 0 else None
            return None
        
        # elif property_type == "rollup":
        #     if property_value and isinstance(property_value, dict):
        #         rollup_type = property_value.get("type", "")
        #         if rollup_type == "number":
        #             value = property_value.get("number", "")
        #             return f"{property_name}: {value} " if value is not None else None
        #         elif rollup_type == "array":
        #             array_length = len(property_value.get("array", []))
        #             return f"{property_name}  [{array_length} items] " if array_length > 0 else None
        #         else:
        #             return f"{property_name}: {rollup_type} " if rollup_type else None
        #     return None
        
        # elif property_type == "formula":
        #     if property_value and isinstance(property_value, dict):
        #         formula_type = property_value.get("type", "")
        #         if formula_type == "string":
        #             string_val = property_value.get('string', '')
        #             return f"{property_name}: {string_val} " if string_val else None
        #         elif formula_type == "number":
        #             number_val = property_value.get('number', '')
        #             return f"{property_name}: {number_val} " if number_val is not None else None
        #         elif formula_type == "boolean":
        #             value = "v" if property_value.get("boolean") else "f"
        #             return f"{property_name}: {value} "
        #         elif formula_type == "date":
        #             date_obj = property_value.get("date")
        #             if date_obj:
        #                 start = date_obj.get("start", "")
        #                 return f"{property_name}: {start} " if start else None
        #         return None
        #     return None
        
        elif property_type == "created_time":
            return f"A tarefa foi criada em {property_value} " if property_value else None
        
        elif property_type == "last_edited_time":
            return f"A tarefa foi editada em {property_value} " if property_value else None
        
        elif property_type == "created_by":
            if property_value and isinstance(property_value, dict):
                if property_value.get("name"):
                    return f"A tarefa foi criada por {property_value['name']}"
                else:
                    user_name = get_notion_username_from_dw(property_value["id"])
                    return f"A tarefa foi criada por {user_name}"
            return None
        
        elif property_type == "last_edited_by":
            if property_value and isinstance(property_value, dict):
                if property_value.get("name"):
                    return f"A tarefa foi atualizada por {property_value['name']}"
                else:
                    user_name = get_notion_username_from_dw(property_value["id"])
                    return f"A tarefa foi atualizada por {user_name}"
            return None
        
        elif property_type == "status":
            if property_value and isinstance(property_value, dict):
                return f"O status {property_name} da tarefa é {property_value['name']}"
            return None
        
        elif property_type == "unique_id":
            if property_value and isinstance(property_value, dict):
                prefix = property_value.get("prefix", "")
                number = property_value.get("number", "")
                if prefix:
                    return f"O identificador {property_name} da tarefa é {prefix}-{number}"
                return f"O identificador {property_name} da tarefa é {number}" if number else None
            return None
        
        # else:
        #     # For unknown property types, return the type only if there's a value
        #     return f"{property_name} [{property_type}]" if property_value else None
    
    def notion_rich_text_to_string(self, rich_text_array: List[Dict[str, Any]]) -> str:
        """Convert Notion rich text array to plain string with markdown formatting."""
        if not rich_text_array:
            return ""
        
        result = ""
        for text_obj in rich_text_array:
            if text_obj["type"] == "text":
                content = text_obj["text"]["content"]
                annotations = text_obj.get("annotations", {})
                
                # # Apply formatting
                # if annotations.get("bold"):
                #     content = f"**{content}**"
                # if annotations.get("italic"):
                #     content = f"*{content}*"
                # if annotations.get("strikethrough"):
                #     content = f"~~{content}~~"
                # if annotations.get("underline"):
                #     content = f"<u>{content}</u>"
                # if annotations.get("code"):
                #     content = f"`{content}`"
                
                # Handle links
                if text_obj["text"].get("link"):
                    url = text_obj["text"]["link"]["url"]
                    #content = f"[{content}]({url})"
                    content = f"{content}"
                
                result += content
            
            elif text_obj["type"] == "mention":
                mention_type = text_obj["mention"]["type"]
                if mention_type == "page":
                    page_name = get_notion_page_name_from_dw(text_obj["mention"]["page"]["id"])
                    result += f"{page_name}"
                elif mention_type == "user":
                    user_name = get_notion_username_from_dw(text_obj["mention"]["user"]["id"])
                    result += f"{user_name}"

                elif mention_type == "date":
                    date_info = text_obj["mention"]["date"]
                    result += f"@{date_info.get('start', 'date')}"
            
            elif text_obj["type"] == "equation":
                expression = text_obj["equation"]["expression"]
                result += f"Código: {expression}"
        
        return result

    async def retrieve_notion_page_content(self, page: NotionPage) -> NotionPage:
        """Get the full text content of a Notion page."""
        if self.notion_client is None:
            raise Exception("Notion client not initialized")
        
        blocks = await self.get_notion_page_blocks(page.page_id)

        comments = await self.get_notion_page_comments(page.page_id)

        properties = await self.get_notion_page_properties(page.page_id)
        
        title_value = next((prop for prop in properties if prop['type'] == 'title'), None)

        page_title = title_value["value"][0]["text"]["content"] if isinstance(title_value["value"], list) else title_value["value"]
        
        # Convert blocks to text
        text_parts = []
        text_parts.insert(0, page_title) 

        # Properties section
        if properties:
            converted_props = [self.notion_property_to_text(prop) for prop in properties if prop and self.notion_property_to_text(prop)]
            if converted_props:
                text_parts.extend(["[Propriedades]"] + converted_props)

        # Comments section  
        if comments:
            converted_comments = [self.notion_comment_to_text(comment) for comment in comments if comment and self.notion_comment_to_text(comment)]
            if converted_comments:
                text_parts.extend(["[Comentários]"] + converted_comments)

        # Blocks section
        if blocks:
            converted_blocks = [self.notion_block_to_text(block) for block in blocks if block and self.notion_block_to_text(block)]
            if converted_blocks:
                text_parts.extend(["[Conteúdo]"] + converted_blocks)
            
        # Join all parts with appropriate spacing
        content = "\n\n".join(text_parts)

        page.content = content,
        page.last_edited_time = next((prop for prop in properties if prop['type'] == 'last_edited_time'), None)["value"]
        
        return page