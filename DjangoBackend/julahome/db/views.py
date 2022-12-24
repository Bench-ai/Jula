import datetime
from django.contrib.auth import get_user_model
from rest_framework.decorators import api_view, permission_classes, authentication_classes
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.authentication import BasicAuthentication
from rest_framework_api_key.permissions import HasAPIKey
from .serializers import UserSerializer, TagClassSerializer
from rest_framework_api_key.models import APIKey


# Create your views here.


@api_view(["PATCH"])
@authentication_classes([BasicAuthentication])
@permission_classes([IsAuthenticated])
def grant_api_key(request):
    data = request.data
    user = str(request.user)
    model_data = get_user_model().objects.get(pk=user)

    if not model_data.is_staff:
        return Response({"invalid": "You are not a member of staff"}, status=400)

    model_data = get_user_model().objects.get(pk=data["req_user"])

    if model_data.api_key:
        return Response({"invalid": "User has already been assigned an apikey"}, status=400)

    api_key, key = APIKey.objects.create_key(name="{}-service-{}".format(data["req_user"],
                                                                         datetime.datetime.now()))

    serializer = UserSerializer(model_data,
                                data={"api_key": "True"},
                                partial=True)

    if serializer.is_valid(raise_exception=True):

        serializer.save()
        ret_data = {"key": key}

        return Response(ret_data, status=201)
    else:
        return Response({"invalid": "missing elements in data"}, status=400)


@api_view(["POST"])
@authentication_classes([BasicAuthentication])
@permission_classes([IsAuthenticated])
def insert_tag_class(request):
    data = request.data

    user = str(request.user)

    print(user)
    model_data = get_user_model().objects.get(pk=user)
    if not model_data.is_staff:
        return Response({"invalid": "You are not a member of staff"}, status=400)

    request.data["user_name"] = user
    serializer = TagClassSerializer(data=request.data)

    if serializer.is_valid(raise_exception=True):

        serializer.save()
        return Response(serializer.data, status=200)
    else:
        return Response({"invalid": "missing elements in data"}, status=400)


@api_view(["POST"])
@permission_classes([HasAPIKey])
def insert_layer(request):
    data = request.data
    file = request.FILES.get("upload")

    print(request.data)
    print(file)

    return Response({"valid": " 0000 missing elements in data"}, status=200)

