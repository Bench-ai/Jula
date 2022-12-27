from django.contrib.auth.base_user import BaseUserManager


class CustomUserManager(BaseUserManager):
    """
    Custom user model manager where email is the unique identifiers
    for authentication instead of usernames.
    """

    def create_user(self, user_name, email, password, **kwargs):
        """
        Create and save a User with the given email and password.
        """

        email = self.normalize_email(email)
        user = self.model(user_name=user_name, email=email, **kwargs)
        user.set_password(password)
        user.save()
        return user

    def create_superuser(self, user_name, email, password, **kwargs):
        """
        Create and save a SuperUser with the given email and password.
        """
        kwargs.setdefault('is_staff', True)
        kwargs.setdefault('is_superuser', True)
        kwargs.setdefault('is_active', True)

        if kwargs.get('is_staff') is not True:
            raise ValueError('Superuser must have is_staff=True.')
        if kwargs.get('is_superuser') is not True:
            raise ValueError('Superuser must have is_superuser=True.')
        return self.create_user(user_name, email, password, **kwargs)
