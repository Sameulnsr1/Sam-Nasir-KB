#!/usr/bin/env python3
"""Build backlinks and MOC pages for the Product Learning vault."""

import json
import os
import re
from collections import defaultdict
from pathlib import Path

VAULT = Path("/Users/sameul.nasir/Documents/Sam Nasir KB/Claude Support/Product Learning")
INDEX = Path("/Users/sameul.nasir/Downloads/lennys-newsletterpodcastdata-all/01-start-here/index.json")

# Tags that are too broad to be useful for relatedness scoring
EXCLUDE_TAGS = {"design", "newsletter", "podcast"}
# Max related items per note
MAX_RELATED = 5

def load_index():
    with open(INDEX) as f:
        data = json.load(f)
    items = []
    for p in data["podcasts"]:
        items.append({
            "title": p["title"],
            "filename": p["filename"],  # e.g. 03-podcasts/boris-cherny.md
            "tags": set(p.get("tags", [])),
            "date": p.get("date", ""),
            "description": p.get("description", ""),
            "guest": p.get("guest", ""),
            "type": "podcast",
            "word_count": p.get("word_count", 0),
        })
    for n in data["newsletters"]:
        items.append({
            "title": n["title"],
            "filename": n["filename"],  # e.g. 02-newsletters/slug.md
            "tags": set(n.get("tags", [])),
            "date": n.get("date", ""),
            "description": n.get("subtitle", ""),
            "type": "newsletter",
            "word_count": n.get("word_count", 0),
            "guest": "",
        })
    return items

def vault_path(item):
    """Convert index filename to vault-relative path."""
    fname = Path(item["filename"]).name  # e.g. boris-cherny.md
    if item["type"] == "podcast":
        return VAULT / "Podcasts" / fname
    else:
        return VAULT / "Newsletters" / fname

def wikilink_name(item):
    """Obsidian wikilink target (no .md extension, relative to Product Learning)."""
    fname = Path(item["filename"]).stem
    if item["type"] == "podcast":
        return f"Podcasts/{fname}"
    else:
        return f"Newsletters/{fname}"

def display_name(item):
    """Display text for wikilinks."""
    if item["guest"]:
        return item["guest"]
    title = item["title"]
    if len(title) > 80:
        title = title[:77] + "..."
    return title

def scan_body_for_guests(items):
    """Scan file bodies for guest name mentions. Returns {guest_name: [mentioned_in_filenames]}."""
    guests = {i["guest"]: i for i in items if i["guest"]}
    mentions = defaultdict(list)  # guest -> list of items that mention them

    for item in items:
        fpath = vault_path(item)
        if not fpath.exists():
            continue
        try:
            body = fpath.read_text(encoding="utf-8").lower()
        except Exception:
            continue
        for guest_name, guest_item in guests.items():
            # Skip self-references
            if item["filename"] == guest_item["filename"]:
                continue
            # Match full name (both parts must appear)
            parts = guest_name.lower().split()
            if len(parts) >= 2 and all(p in body for p in parts):
                mentions[guest_name].append(item)

    return mentions

def compute_related(items):
    """Compute related items for each file based on tag overlap + guest mentions."""
    guest_mentions = scan_body_for_guests(items)
    
    # Build guest -> items index
    guest_items = defaultdict(list)
    for item in items:
        if item["guest"]:
            guest_items[item["guest"]].append(item)

    related = {}  # filename -> list of (score, item)
    
    for i, item in enumerate(items):
        scores = []
        item_tags = item["tags"] - EXCLUDE_TAGS
        
        for j, other in enumerate(items):
            if i == j:
                continue
            other_tags = other["tags"] - EXCLUDE_TAGS
            
            # Tag overlap score
            shared = len(item_tags & other_tags)
            if shared < 2:
                continue
            
            score = shared
            
            # Boost if guest is mentioned in the other file
            if item["guest"] and item in guest_mentions.get(item["guest"], []):
                score += 3
            if other["guest"] and other in guest_mentions.get(other["guest"], []):
                score += 3
                
            scores.append((score, other))
        
        # Sort by score desc, then date desc
        scores.sort(key=lambda x: (x[0], x[1]["date"]), reverse=True)
        related[item["filename"]] = [s[1] for s in scores[:MAX_RELATED]]
    
    return related, guest_mentions

def generate_topic_mocs(items):
    """Generate Topic MOC pages."""
    topics_dir = VAULT / "Topics"
    topics_dir.mkdir(exist_ok=True)
    
    tag_items = defaultdict(list)
    for item in items:
        for tag in item["tags"]:
            if tag not in EXCLUDE_TAGS:
                tag_items[tag].append(item)
    
    tag_display = {
        "ai": "AI",
        "analytics": "Analytics",
        "b2b": "B2B",
        "b2c": "B2C",
        "career": "Career",
        "engineering": "Engineering",
        "go-to-market": "Go-to-Market",
        "growth": "Growth",
        "leadership": "Leadership",
        "organization": "Organization",
        "pricing": "Pricing",
        "product-management": "Product Management",
        "startups": "Startups",
        "strategy": "Strategy",
    }
    
    for tag, tag_items_list in tag_items.items():
        name = tag_display.get(tag, tag.title())
        tag_items_list.sort(key=lambda x: x["date"], reverse=True)
        
        podcasts = [i for i in tag_items_list if i["type"] == "podcast"]
        newsletters = [i for i in tag_items_list if i["type"] == "newsletter"]
        
        lines = [
            f"# {name}",
            "",
            f"> {len(tag_items_list)} articles and transcripts on **{name.lower()}**.",
            "",
            f"**Podcasts**: {len(podcasts)} · **Newsletters**: {len(newsletters)}",
            "",
        ]
        
        # Related topics - find tags that co-occur frequently
        co_tags = defaultdict(int)
        for item in tag_items_list:
            for t in item["tags"] - EXCLUDE_TAGS - {tag}:
                co_tags[t] += 1
        top_co = sorted(co_tags.items(), key=lambda x: x[1], reverse=True)[:5]
        if top_co:
            related_links = " · ".join(f"[[Topics/{tag_display.get(t, t.title())}|{tag_display.get(t, t.title())}]]" for t, _ in top_co)
            lines.extend(["**Related topics**: " + related_links, ""])
        
        lines.append("---")
        lines.append("")
        
        if podcasts:
            lines.append("## Podcasts")
            lines.append("")
            for p in podcasts[:50]:  # Cap for readability
                link = f"[[{wikilink_name(p)}|{display_name(p)}]]"
                lines.append(f"- {p['date']} — {link}")
            if len(podcasts) > 50:
                lines.append(f"- *...and {len(podcasts) - 50} more*")
            lines.append("")
        
        if newsletters:
            lines.append("## Newsletters")
            lines.append("")
            for n in newsletters[:50]:
                link = f"[[{wikilink_name(n)}|{display_name(n)}]]"
                lines.append(f"- {n['date']} — {link}")
            if len(newsletters) > 50:
                lines.append(f"- *...and {len(newsletters) - 50} more*")
            lines.append("")
        
        fpath = topics_dir / f"{name}.md"
        fpath.write_text("\n".join(lines), encoding="utf-8")
        print(f"  Created Topic MOC: {name} ({len(tag_items_list)} items)")

def generate_guest_pages(items, guest_mentions):
    """Generate Guest pages for guests with cross-references."""
    guests_dir = VAULT / "Guests"
    guests_dir.mkdir(exist_ok=True)
    
    # Build guest -> podcast item mapping
    guest_podcast = {}
    for item in items:
        if item["guest"]:
            guest_podcast[item["guest"]] = item
    
    created = 0
    for guest_name, podcast_item in guest_podcast.items():
        mentions = guest_mentions.get(guest_name, [])
        
        # Only create pages for guests mentioned in other content
        if not mentions:
            continue
        
        lines = [
            f"# {guest_name}",
            "",
        ]
        
        if podcast_item["description"]:
            lines.extend([f"> {podcast_item['description']}", ""])
        
        # Tags
        tags = podcast_item["tags"] - EXCLUDE_TAGS
        if tags:
            tag_display = {
                "ai": "AI", "analytics": "Analytics", "b2b": "B2B", "b2c": "B2C",
                "career": "Career", "engineering": "Engineering", "go-to-market": "Go-to-Market",
                "growth": "Growth", "leadership": "Leadership", "organization": "Organization",
                "pricing": "Pricing", "product-management": "Product Management",
                "startups": "Startups", "strategy": "Strategy",
            }
            tag_links = " · ".join(f"[[Topics/{tag_display.get(t, t.title())}|{tag_display.get(t, t.title())}]]" for t in sorted(tags))
            lines.extend([f"**Topics**: {tag_links}", ""])
        
        lines.extend(["---", ""])
        
        # Podcast episode
        lines.append("## Podcast Episode")
        lines.append("")
        link = f"[[{wikilink_name(podcast_item)}|{podcast_item['title']}]]"
        lines.append(f"- {podcast_item['date']} — {link}")
        lines.append("")
        
        # Mentioned in
        if mentions:
            lines.append(f"## Mentioned In ({len(mentions)} articles)")
            lines.append("")
            mentions.sort(key=lambda x: x["date"], reverse=True)
            for m in mentions[:20]:
                link = f"[[{wikilink_name(m)}|{display_name(m)}]]"
                type_label = "🎙️" if m["type"] == "podcast" else "📰"
                lines.append(f"- {type_label} {m['date']} — {link}")
            if len(mentions) > 20:
                lines.append(f"- *...and {len(mentions) - 20} more*")
            lines.append("")
        
        # Write file
        safe_name = re.sub(r'[<>:"/\\|?*]', '', guest_name)
        fpath = guests_dir / f"{safe_name}.md"
        fpath.write_text("\n".join(lines), encoding="utf-8")
        created += 1
    
    print(f"  Created {created} Guest pages")
    return created

def append_related_sections(items, related, guest_mentions):
    """Append ## Related section to each content file."""
    tag_display = {
        "ai": "AI", "analytics": "Analytics", "b2b": "B2B", "b2c": "B2C",
        "career": "Career", "engineering": "Engineering", "go-to-market": "Go-to-Market",
        "growth": "Growth", "leadership": "Leadership", "organization": "Organization",
        "pricing": "Pricing", "product-management": "Product Management",
        "startups": "Startups", "strategy": "Strategy",
    }
    
    updated = 0
    skipped = 0
    
    for item in items:
        fpath = vault_path(item)
        if not fpath.exists():
            skipped += 1
            continue
        
        try:
            content = fpath.read_text(encoding="utf-8")
        except Exception:
            skipped += 1
            continue
        
        # Remove existing Related section if present (idempotent)
        content = re.sub(r'\n---\n\n## Related\n.*', '', content, flags=re.DOTALL)
        content = content.rstrip()
        
        # Build Related section
        sections = []
        
        # Topic links
        tags = item["tags"] - EXCLUDE_TAGS
        if tags:
            tag_links = " · ".join(
                f"[[Topics/{tag_display.get(t, t.title())}|{tag_display.get(t, t.title())}]]"
                for t in sorted(tags)
            )
            sections.append(f"**Topics**: {tag_links}")
        
        # Guest link
        if item["guest"]:
            safe_name = re.sub(r'[<>:"/\\|?*]', '', item["guest"])
            sections.append(f"**Guest**: [[Guests/{safe_name}|{item['guest']}]]")
        
        # Related content
        rel_items = related.get(item["filename"], [])
        if rel_items:
            see_also = []
            for r in rel_items:
                link = f"[[{wikilink_name(r)}|{display_name(r)}]]"
                type_label = "🎙️" if r["type"] == "podcast" else "📰"
                see_also.append(f"- {type_label} {link}")
            sections.append("**See also**:\n" + "\n".join(see_also))
        
        if sections:
            related_block = "\n---\n\n## Related\n\n" + "\n\n".join(sections) + "\n"
            content += related_block
            fpath.write_text(content, encoding="utf-8")
            updated += 1
    
    print(f"  Updated {updated} files with Related sections (skipped {skipped})")

def update_overview(items):
    """Update Overview.md with Topic MOC links."""
    overview = VAULT / "Overview.md"
    content = overview.read_text(encoding="utf-8")
    
    # Add Topics section if not present
    if "## Topic Maps" not in content:
        topic_section = """
---

## Topic Maps

Browse by topic — each page links to all related podcasts and newsletters:

| Topic | Articles |
|---|---|
| [[Topics/AI|AI]] | 128 |
| [[Topics/Analytics|Analytics]] | 107 |
| [[Topics/B2B|B2B]] | 184 |
| [[Topics/B2C|B2C]] | 163 |
| [[Topics/Career|Career]] | 247 |
| [[Topics/Engineering|Engineering]] | 175 |
| [[Topics/Go-to-Market|Go-to-Market]] | 107 |
| [[Topics/Growth|Growth]] | 289 |
| [[Topics/Leadership|Leadership]] | 493 |
| [[Topics/Organization|Organization]] | 49 |
| [[Topics/Pricing|Pricing]] | 52 |
| [[Topics/Product Management|Product Management]] | 191 |
| [[Topics/Startups|Startups]] | 264 |
| [[Topics/Strategy|Strategy]] | 364 |

See also: [[Guests/]] for pages on individual podcast guests.
"""
        content = content.rstrip() + "\n" + topic_section
        overview.write_text(content, encoding="utf-8")
        print("  Updated Overview.md with Topic Maps")

def main():
    print("Loading index...")
    items = load_index()
    print(f"  {len(items)} items ({sum(1 for i in items if i['type']=='podcast')} podcasts, {sum(1 for i in items if i['type']=='newsletter')} newsletters)")
    
    print("\nScanning for guest mentions and computing relationships...")
    related, guest_mentions = compute_related(items)
    print(f"  {len(guest_mentions)} guests mentioned in other content")
    
    print("\nGenerating Topic MOC pages...")
    generate_topic_mocs(items)
    
    print("\nGenerating Guest pages...")
    generate_guest_pages(items, guest_mentions)
    
    print("\nAppending Related sections to content files...")
    append_related_sections(items, related, guest_mentions)
    
    print("\nUpdating Overview.md...")
    update_overview(items)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
