from uuid import uuid4
from django.db import models
from django.contrib.auth.models import (AbstractBaseUser)
from django.contrib.auth.models import PermissionsMixin
from .managers import CustomUserManager


class User(AbstractBaseUser, PermissionsMixin):
    user_name = models.CharField(max_length=100,
                                 help_text="the name of the layer",
                                 primary_key=True)

    email = models.EmailField(
        max_length=255,
        unique=True,
    )

    api_key = models.CharField(max_length=255,
                               help_text="the name of the layer",
                               unique=True,
                               null=True)

    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)  # a admin user; non super-user

    creation_timestamp = models.DateTimeField(auto_now_add=True)

    update_timestamp = models.DateTimeField(auto_now=True)

    is_deleted = models.BooleanField(default=False, null=False, blank=False)

    # notice the absence of a "Password field", that is built in.

    USERNAME_FIELD = "user_name"
    REQUIRED_FIELDS = ["email"]  # Email & Password are required by default.

    def __str__(self):
        return self.user_name

    objects = CustomUserManager()
