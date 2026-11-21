from pydantic import BaseModel, Field


class Experience(BaseModel):
    title: str
    company: str = ""
    period: str = ""


class Education(BaseModel):
    degree: str
    school: str = ""
    year: str = ""


class Resume(BaseModel):
    name: str = ""
    email: str = ""
    phone: str = ""
    headline: str = ""
    summary: str = ""
    skills: list[str] = Field(default_factory=list)
    experience: list[Experience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)

    def to_text(self) -> str:
        parts = [self.headline, self.summary, "Skills: " + ", ".join(self.skills)]
        parts += [f"{e.title} at {e.company}" for e in self.experience]
        parts += [f"{e.degree}, {e.school}" for e in self.education]
        return "\n".join(p for p in parts if p)
